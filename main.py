import numpy as np
from pprint import pp
import pandas as pd
from collections import defaultdict
from numba import jit
import numpy.typing as npt

class ChoiceCost:
    def __init__(self, base_cost, per_person_cost):
        self.base_cost = base_cost
        self.per_person_cost = per_person_cost

    def calculate_cost(self, family_size):
        return self.base_cost + self.per_person_cost * family_size
    

class FamilyData:
    def __init__(self, family_size, family_choices):
        self.family_size = family_size
        self.family_choices = family_choices

    def get_family_size(self):
        return self.family_size

    def get_family_choices(self):
        return self.family_choices
    

@jit(nopython=True)
def calculate_accounting_penalty(daily_occupancy: npt.NDArray) -> float:
    """Vectorized and JIT-compiled accounting penalty calculation."""
    N_d = daily_occupancy[1:]
    N_d_next = np.roll(N_d, 1)
    N_d_next[0] = N_d[0]
    
    diff = N_d - 125
    penalty = 0.0
    for i in range(len(diff)):
        if diff[i] > 0:
            penalty += (diff[i] / 400.0) * (N_d[i] ** (0.5 + abs(N_d[i] - N_d_next[i]) / 50.0))
    return penalty

class ParticleSwarmOptimizer:
    def __init__(self, n_particles, n_families, n_days, family_data):
        self.n_particles = n_particles
        self.n_families = n_families
        self.n_days = n_days
        
        # Pre-calculate and cache family data
        self.family_size = np.array([family.get_family_size() for family in family_data])
        self.family_choices = np.array([family.get_family_choices() for family in family_data])
        self.total_people = np.sum(self.family_size)
        self.avg_daily_people = self.total_people / self.n_days
        
        # Create choice sets and preference matrices for faster lookup
        self.family_choice_sets = [set(choices) for choices in self.family_choices]
        self.choice_matrix = np.zeros((n_families, n_days + 1))
        for i, choices in enumerate(self.family_choices):
            self.choice_matrix[i, choices] = 1
        
        # Cache family indices by size for faster repair
        self.families_by_size = defaultdict(list)
        for i, size in enumerate(self.family_size):
            self.families_by_size[size].append(i)
        
        self.min_people = 125
        self.max_people = 300
        
        self.choice_costs = {
            0: ChoiceCost(0, 0),
            1: ChoiceCost(50, 0),
            2: ChoiceCost(50, 9),
            3: ChoiceCost(100, 9),
            4: ChoiceCost(200, 9),
            5: ChoiceCost(200, 18),
            6: ChoiceCost(300, 18),
            7: ChoiceCost(300, 36),
            8: ChoiceCost(400, 36),
            9: ChoiceCost(500, 36+199),
            -1: ChoiceCost(500, 36+398)
        }
        
        # Initialize particles with improved strategy
        self.particles = self.init_particles()
        self.velocities = np.zeros((n_particles, n_families))
        
        self.p_best = self.particles.copy()
        self.p_best_scores = np.full(n_particles, float('inf'))
        self.g_best = self.particles[0].copy()
        self.g_best_score = float('inf')
        
        # Adaptive parameters
        self.w_max = 0.9
        self.w_min = 0.2
        self.w = self.w_max
        self.c1_start = 2.5
        self.c1_end = 0.5
        self.c2_start = 0.5
        self.c2_end = 2.5
        
        # Local neighborhood topology
        self.neighborhood_size = 5
        self.l_best = self.particles.copy()
        self.l_best_scores = np.full(n_particles, float('inf'))
        
        # Pre-compute choice costs for each family size
        self.choice_cost_cache = {}
        max_family_size = max(self.family_size)
        for choice in range(-1, 10):
            self.choice_cost_cache[choice] = np.array([
                self.choice_costs[choice].calculate_cost(size)
                for size in range(max_family_size + 1)
            ])
        
        # Create lookup arrays for faster access
        self.choice_lookup = -np.ones((n_families, n_days + 1), dtype=np.int8)
        for i, choices in enumerate(self.family_choices):
            for j, choice in enumerate(choices):
                self.choice_lookup[i, choice] = j

    def init_particles(self):
        """Initialize particles with improved strategy using choice preferences and load balancing."""
        particles = np.zeros((self.n_particles, self.n_families), dtype=int)
        
        # Initialize multiple particles with different strategies
        for p in range(self.n_particles):
            if p < self.n_particles // 3:
                # Strategy 1: Prioritize first choices and balance load
                particles[p] = self._init_with_first_choices()
            elif p < 2 * self.n_particles // 3:
                # Strategy 2: Random assignment with preference weights
                particles[p] = self._init_with_weighted_choices()
            else:
                # Strategy 3: Load-balanced assignment
                particles[p] = self._init_with_load_balance()
            
            # Repair if needed
            daily_occupancy = np.zeros(self.n_days + 1, dtype=int)
            np.add.at(daily_occupancy, particles[p], self.family_size)
            if np.any((daily_occupancy[1:] < self.min_people) | (daily_occupancy[1:] > self.max_people)):
                particles[p] = self._repair_schedule_fast(particles[p], daily_occupancy)
        
        return particles

    def _init_with_first_choices(self):
        """Initialize focusing on first choices while maintaining balance."""
        schedule = np.zeros(self.n_families, dtype=int)
        daily_occupancy = np.zeros(self.n_days + 1, dtype=int)
        
        # Sort families by size and preference flexibility
        family_order = sorted(
            range(self.n_families),
            key=lambda f: (-self.family_size[f], len(set(self.family_choices[f])))
        )
        
        for family_id in family_order:
            choices = self.family_choices[family_id]
            family_size = self.family_size[family_id]
            
            # Calculate scores for each choice
            choice_scores = []
            for choice in choices:
                if daily_occupancy[choice] + family_size <= self.max_people:
                    # Score based on current occupancy and choice rank
                    occupancy_score = 1 - (daily_occupancy[choice] / self.max_people)
                    choice_rank = np.where(choices == choice)[0][0]
                    preference_score = 1 / (1 + choice_rank)
                    total_score = occupancy_score * 0.7 + preference_score * 0.3
                    choice_scores.append((total_score, choice))
            
            if choice_scores:
                # Choose the best scoring day
                best_score, best_day = max(choice_scores)
                schedule[family_id] = best_day
                daily_occupancy[best_day] += family_size
            else:
                # If no preferred day works, find any day with space
                available_days = []
                for day in range(1, self.n_days + 1):
                    if daily_occupancy[day] + family_size <= self.max_people:
                        occupancy_score = 1 - (daily_occupancy[day] / self.max_people)
                        available_days.append((occupancy_score, day))
                
                if available_days:
                    _, best_day = max(available_days)
                    schedule[family_id] = best_day
                    daily_occupancy[best_day] += family_size
                else:
                    # Last resort: find day with minimum overflow
                    min_overflow = float('inf')
                    best_day = 1
                    for day in range(1, self.n_days + 1):
                        overflow = daily_occupancy[day] + family_size - self.max_people
                        if overflow < min_overflow:
                            min_overflow = overflow
                            best_day = day
                    schedule[family_id] = best_day
                    daily_occupancy[best_day] += family_size
        
        return schedule

    def _init_with_weighted_choices(self):
        """Initialize using weighted random choices based on preferences."""
        schedule = np.zeros(self.n_families, dtype=int)
        daily_occupancy = np.zeros(self.n_days + 1, dtype=int)
        
        # Define preference weights with stronger bias towards top choices
        weights = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0125, 0.0125])
        
        # Sort families by size (largest first)
        family_order = np.argsort(-self.family_size)
        
        for family_id in family_order:
            choices = self.family_choices[family_id]
            family_size = self.family_size[family_id]
            
            # Calculate occupancy factor with stronger penalty for high occupancy
            occupancy_factor = np.exp(-np.square(daily_occupancy[1:] - self.avg_daily_people) / (2 * self.avg_daily_people))
            
            # Try multiple times with different strategies
            assigned = False
            for strategy in range(3):
                if strategy == 0:
                    # First try: Use choice weights and occupancy
                    choice_weights = weights[:len(choices)] * occupancy_factor[choices - 1]
                elif strategy == 1:
                    # Second try: Focus more on occupancy
                    choice_weights = occupancy_factor[choices - 1]
                else:
                    # Third try: Focus more on preferences
                    choice_weights = weights[:len(choices)]
                
                choice_weights = np.maximum(choice_weights, 1e-10)  # Avoid zero probabilities
                choice_weights /= choice_weights.sum()
                
                # Try a few times with current strategy
                for _ in range(3):
                    day = choices[np.random.choice(len(choices), p=choice_weights)]
                    if daily_occupancy[day] + family_size <= self.max_people:
                        schedule[family_id] = day
                        daily_occupancy[day] += family_size
                        assigned = True
                        break
                
                if assigned:
                    break
            
            if not assigned:
                # If still not assigned, find any day with minimum overflow
                min_overflow = float('inf')
                best_day = 1
                for day in range(1, self.n_days + 1):
                    overflow = max(0, daily_occupancy[day] + family_size - self.max_people)
                    if overflow < min_overflow:
                        min_overflow = overflow
                        best_day = day
                schedule[family_id] = best_day
                daily_occupancy[best_day] += family_size
        
        return schedule

    def _init_with_load_balance(self):
        """Initialize focusing on load balancing."""
        schedule = np.zeros(self.n_families, dtype=int)
        daily_occupancy = np.zeros(self.n_days + 1, dtype=int)
        
        # Calculate target occupancy range
        target_min = max(self.min_people, self.avg_daily_people * 0.9)
        target_max = min(self.max_people, self.avg_daily_people * 1.1)
        
        def get_day_score(day, family_size, is_preferred):
            current_occupancy = daily_occupancy[day]
            if current_occupancy + family_size > self.max_people:
                return float('-inf')
            
            # Score based on how close to target range
            if current_occupancy < target_min:
                balance_score = 1 - (target_min - current_occupancy) / target_min
            elif current_occupancy > target_max:
                balance_score = 1 - (current_occupancy - target_max) / (self.max_people - target_max)
            else:
                balance_score = 1.0
            
            # Bonus for preferred days
            preference_bonus = 0.3 if is_preferred else 0
            
            return balance_score + preference_bonus
        
        # Sort families by size and preference flexibility
        family_order = sorted(
            range(self.n_families),
            key=lambda f: (-self.family_size[f], len(set(self.family_choices[f])))
        )
        
        for family_id in family_order:
            family_size = self.family_size[family_id]
            choices = set(self.family_choices[family_id])
            
            # Score all possible days
            day_scores = []
            for day in range(1, self.n_days + 1):
                score = get_day_score(day, family_size, day in choices)
                if score > float('-inf'):
                    day_scores.append((score, day))
            
            if day_scores:
                # Choose the best scoring day
                _, best_day = max(day_scores)
                schedule[family_id] = best_day
                daily_occupancy[best_day] += family_size
            else:
                # If no day works, find the one with minimum overflow
                min_overflow = float('inf')
                best_day = 1
                for day in range(1, self.n_days + 1):
                    overflow = daily_occupancy[day] + family_size - self.max_people
                    if overflow < min_overflow:
                        min_overflow = overflow
                        best_day = day
                schedule[family_id] = best_day
                daily_occupancy[best_day] += family_size
        
        return schedule

    def _repair_schedule_fast(self, schedule, daily_occupancy):
        """Optimized repair mechanism."""
        # Handle underflow
        underflow_days = np.where(daily_occupancy[1:] < self.min_people)[0] + 1
        if len(underflow_days) > 0:
            # Sort by deficit for better repair order
            deficits = self.min_people - daily_occupancy[underflow_days]
            underflow_days = underflow_days[np.argsort(-deficits)]
            
            # Pre-calculate source days once
            source_mask = daily_occupancy[1:] > self.min_people + 20
            source_days = np.where(source_mask)[0] + 1
            source_excess = daily_occupancy[source_days] - self.min_people
            source_order = np.argsort(-source_excess)
            source_days = source_days[source_order]
            
            for day in underflow_days:
                needed = self.min_people - daily_occupancy[day]
                if needed <= 0:
                    continue
                
                # Find suitable families in batches
                for source_day in source_days:
                    if daily_occupancy[source_day] <= self.min_people:
                        continue
                    
                    # Get families that can be moved
                    families_in_day = np.where(schedule == source_day)[0]
                    if len(families_in_day) == 0:
                        continue
                    
                    # Sort by size and preference
                    families_in_day = sorted(
                        families_in_day,
                        key=lambda f: (
                            -self.family_size[f] if self.family_size[f] <= needed else -needed,
                            day in self.family_choice_sets[f]
                        )
                    )
                    
                    # Move families
                    for family_id in families_in_day:
                        size = self.family_size[family_id]
                        if size > needed:
                            continue
                        
                        if (daily_occupancy[source_day] - size >= self.min_people and
                            daily_occupancy[day] + size <= self.max_people):
                            schedule[family_id] = day
                            daily_occupancy[source_day] -= size
                            daily_occupancy[day] += size
                            needed -= size
                            
                            if needed <= 0:
                                break
                    
                    if needed <= 0:
                        break
        
        # Handle overflow similarly
        overflow_days = np.where(daily_occupancy[1:] > self.max_people)[0] + 1
        if len(overflow_days) > 0:
            # Sort by excess
            excess = daily_occupancy[overflow_days] - self.max_people
            overflow_days = overflow_days[np.argsort(-excess)]
            
            # Pre-calculate target days
            target_mask = daily_occupancy[1:] < self.max_people - 20
            target_days = np.where(target_mask)[0] + 1
            available_space = self.max_people - daily_occupancy[target_days]
            target_order = np.argsort(-available_space)
            target_days = target_days[target_order]
            
            for day in overflow_days:
                excess = daily_occupancy[day] - self.max_people
                if excess <= 0:
                    continue
                
                # Get and sort families
                families_in_day = np.where(schedule == day)[0]
                families_in_day = sorted(
                    families_in_day,
                    key=lambda f: (
                        self.family_size[f],
                        day not in self.family_choice_sets[f]
                    )
                )
                
                for family_id in families_in_day:
                    size = self.family_size[family_id]
                    moved = False
                    
                    # Try preferred days first
                    for target_day in self.family_choices[family_id]:
                        if target_day not in target_days:
                            continue
                        
                        if (daily_occupancy[day] - size >= self.min_people and
                            daily_occupancy[target_day] + size <= self.max_people):
                            schedule[family_id] = target_day
                            daily_occupancy[day] -= size
                            daily_occupancy[target_day] += size
                            excess -= size
                            moved = True
                            break
                    
                    if not moved:
                        # Try any available day
                        for target_day in target_days:
                            if (daily_occupancy[day] - size >= self.min_people and
                                daily_occupancy[target_day] + size <= self.max_people):
                                schedule[family_id] = target_day
                                daily_occupancy[day] -= size
                                daily_occupancy[target_day] += size
                                excess -= size
                                break
                    
                    if excess <= 0:
                        break
        
        return schedule

    def _local_search(self, schedule):
        """Optimized local search."""
        daily_occupancy = np.zeros(self.n_days + 1, dtype=np.int32)
        np.add.at(daily_occupancy, schedule, self.family_size)
        
        # Pre-calculate current costs
        current_costs = np.zeros(self.n_families)
        for family_id in range(self.n_families):
            current_day = schedule[family_id]
            choice = self.choice_lookup[family_id, current_day]
            current_costs[family_id] = self.choice_cost_cache[max(-1, choice)][self.family_size[family_id]]
        
        # Try to improve in batches
        improved = False
        for _ in range(5):  # Limit iterations for speed
            # Select random subset of families
            family_subset = np.random.choice(self.n_families, size=min(500, self.n_families), replace=False)
            
            for family_id in family_subset:
                current_day = schedule[family_id]
                family_size = self.family_size[family_id]
                current_cost = current_costs[family_id]
                
                # Try moving to better days
                for new_day in self.family_choices[family_id]:
                    if new_day == current_day:
                        continue
                    
                    new_occupancy_from = daily_occupancy[current_day] - family_size
                    new_occupancy_to = daily_occupancy[new_day] + family_size
                    
                    if (new_occupancy_from >= self.min_people and 
                        new_occupancy_to <= self.max_people):
                        choice = self.choice_lookup[family_id, new_day]
                        new_cost = self.choice_cost_cache[choice][family_size]
                        
                        if new_cost < current_cost:
                            schedule[family_id] = new_day
                            daily_occupancy[current_day] = new_occupancy_from
                            daily_occupancy[new_day] = new_occupancy_to
                            current_costs[family_id] = new_cost
                            improved = True
                            break
        
        return schedule

    def evaluate_fitness(self, schedule):
        """Optimized fitness evaluation."""
        daily_occupancy = np.zeros(self.n_days + 1, dtype=np.int32)
        np.add.at(daily_occupancy, schedule, self.family_size)
        
        if np.any((daily_occupancy[1:] < self.min_people) | (daily_occupancy[1:] > self.max_people)):
            return float('inf')
        
        # Calculate preference costs using cached values
        total_cost = 0
        for family_id, assigned_day in enumerate(schedule):
            choice = self.choice_lookup[family_id, assigned_day]
            family_size = self.family_size[family_id]
            if choice >= 0:
                total_cost += self.choice_cost_cache[choice][family_size]
            else:
                total_cost += self.choice_cost_cache[-1][family_size]
        
        # Calculate accounting penalty using JIT-compiled function
        accounting_penalty = calculate_accounting_penalty(daily_occupancy)
        
        return total_cost + accounting_penalty

    def update(self):
        """Optimized update mechanism."""
        # Update adaptive parameters
        progress = self.iteration / self.max_iterations
        self.w = self.w_max - (self.w_max - self.w_min) * progress
        c1 = self.c1_start + (self.c1_end - self.c1_start) * progress
        c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
        
        # Update velocities with improved exploration
        r1, r2 = np.random.rand(2, self.n_particles, self.n_families)
        
        # Add random perturbation for exploration
        perturbation = np.random.normal(0, 1, (self.n_particles, self.n_families)) * (1 - progress)
        
        self.velocities = (
            self.w * self.velocities +
            c1 * r1 * (self.p_best - self.particles) +
            c2 * r2 * (self.g_best - self.particles) +
            perturbation
        )
        
        self.velocities = np.clip(self.velocities, -5, 5)
        new_positions = np.clip(np.round(self.particles + self.velocities), 1, self.n_days).astype(int)
        
        # Process each particle
        for i in range(self.n_particles):
            daily_occupancy = np.zeros(self.n_days + 1, dtype=np.int32)
            np.add.at(daily_occupancy, new_positions[i], self.family_size)
            
            if np.any((daily_occupancy[1:] < self.min_people) | (daily_occupancy[1:] > self.max_people)):
                new_positions[i] = self._repair_schedule_fast(new_positions[i], daily_occupancy)
            
            # Apply local search periodically
            if np.random.random() < 0.1:  # 10% chance
                new_positions[i] = self._local_search(new_positions[i])
            
            # Enhanced mutation strategy
            if np.random.random() < 0.2 * (1 - progress):
                # Vectorized mutation
                n_mutations = max(1, int(10 * (1 - progress)))
                mutation_mask = np.random.random(self.n_families) < 0.1
                families_to_mutate = np.where(mutation_mask)[0]
                
                if len(families_to_mutate) > 0:
                    for family_id in families_to_mutate:
                        if np.random.random() < 0.8:
                            choices = self.family_choices[family_id]
                            weights = np.array([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025])[:len(choices)]
                            weights /= weights.sum()
                            new_positions[i, family_id] = choices[np.random.choice(len(choices), p=weights)]
                        else:
                            new_positions[i, family_id] = np.random.randint(1, self.n_days + 1)
                    
                    # Repair after batch mutation
                    daily_occupancy = np.zeros(self.n_days + 1, dtype=np.int32)
                    np.add.at(daily_occupancy, new_positions[i], self.family_size)
                    if np.any((daily_occupancy[1:] < self.min_people) | (daily_occupancy[1:] > self.max_people)):
                        new_positions[i] = self._repair_schedule_fast(new_positions[i], daily_occupancy)
            
            score = self.evaluate_fitness(new_positions[i])
            
            # Update personal best with simulated annealing acceptance
            if score < self.p_best_scores[i]:
                self.p_best[i] = new_positions[i].copy()
                self.p_best_scores[i] = score
                
                # Update global best
                if score < self.g_best_score:
                    self.g_best = new_positions[i].copy()
                    self.g_best_score = score
            elif score < float('inf'):
                # Simulated annealing with adaptive temperature
                temperature = max(self.g_best_score * 0.001, self.g_best_score * (1 - progress))
                acceptance_prob = np.exp(-(score - self.p_best_scores[i]) / temperature)
                if np.random.random() < acceptance_prob:
                    self.p_best[i] = new_positions[i].copy()
                    self.p_best_scores[i] = score
        
        self.particles = new_positions
        self.iteration += 1


def solve_santa_workshop():
    n_particles = 300  # Increased population size
    n_families = 5000
    n_days = 100
    max_iterations = 2000  # Increased iterations for better exploration
    
    family_data = pd.read_csv('data/family_data.csv')
    
    families = []
    for _, row in family_data.iterrows():
        choices = [row[f'choice_{i}'] for i in range(10)]
        n_people = row['n_people']
        families.append(FamilyData(n_people, choices))
    
    pso = ParticleSwarmOptimizer(n_particles, n_families, n_days, families)
    pso.max_iterations = max_iterations
    pso.iteration = 0
    
    best_score = float('inf')
    stagnation_counter = 0
    stagnation_threshold = 50  # Increased threshold
    
    for iteration in range(max_iterations):
        pso.update()
        
        if pso.g_best_score < best_score:
            best_score = pso.g_best_score
            stagnation_counter = 0
            print(f"New best score at iteration {iteration}: {best_score}")
        else:
            stagnation_counter += 1
        
        if stagnation_counter > stagnation_threshold:
            print(f"Reinitializing 50% of particles at iteration {iteration}")
            new_particles = pso.init_particles()
            replace_idx = np.random.choice(n_particles, n_particles // 2, replace=False)
            pso.particles[replace_idx] = new_particles[replace_idx]
            pso.velocities[replace_idx] = 0
            stagnation_counter = 0
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Score: {pso.g_best_score}")
    
    return pso.g_best, pso.g_best_score

if __name__ == "__main__":
    best_schedule, best_score = solve_santa_workshop()
    print(f"Final best score: {best_score}")
    
    # Save results
    np.save('best_schedule.npy', best_schedule)
    print("Schedule saved to best_schedule.npy")
