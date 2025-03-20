import numpy as np
from pprint import pp
import pandas as pd

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
    

class ParticleSwarmOptimizer:
    def __init__(self, n_particles, n_families, n_days, family_data):
        self.n_particles = n_particles
        self.n_families = n_families
        self.n_days = n_days
        
        self.family_size = np.array([family.get_family_size() for family in family_data])
        self.family_choices = np.array([family.get_family_choices() for family in family_data])
        
        self.min_people = 125
        self.max_people = 300
        
        self.choice_costs = {
            0: ChoiceCost(0, 0),         # choice_0: no cost
            1: ChoiceCost(50, 0),        # choice_1: $50 gift card
            2: ChoiceCost(50, 9),        # choice_2: $50 gift card + 25% off buffet ($9 per person)
            3: ChoiceCost(100, 9),       # choice_3: $100 gift card + 25% off buffet
            4: ChoiceCost(200, 9),       # choice_4: $200 gift card + 25% off buffet
            5: ChoiceCost(200, 18),      # choice_5: $200 gift card + 50% off buffet ($18 per person)
            6: ChoiceCost(300, 18),      # choice_6: $300 gift card + 50% off buffet
            7: ChoiceCost(300, 36),      # choice_7: $300 gift card + free buffet ($36 per person)
            8: ChoiceCost(400, 36),      # choice_8: $400 gift card + free buffet
            9: ChoiceCost(500, 36+199),  # choice_9: $500 gift card + free buffet + 50% off helicopter ($36 + $199 per person)
            -1: ChoiceCost(500, 36+398)  # otherwise: $500 gift card + free buffet + free helicopter ($36 + $398 per person)
        }
        
        self.particles = self.init_particles()
        self.velocities = np.zeros((n_particles, n_families))
        
        self.p_best = self.particles.copy()
        self.p_best_scores = np.full(n_particles, float('inf'))
        self.g_best = self.particles[0].copy()
        self.g_best_score = float('inf')
        
        self.w = 0.5  # inertia weight
        self.c1 = 1.0  # cognitive weight
        self.c2 = 2.0  # social weight

    def init_particles(self):
        """Initialize particles with valid solutions meeting capacity constraints."""
        particles = np.zeros((self.n_particles, self.n_families), dtype=int)
        
        choice_weights = np.full((self.n_families, self.n_days), 0.2 / self.n_days)
        for family_id in range(self.n_families):
            choices = [c-1 for c in self.family_choices[family_id]]  # Convert to 0-based indexing
            choice_weights[family_id, choices] = 0.8 / len(choices)
            choice_weights[family_id] /= choice_weights[family_id].sum()
        
        for p in range(self.n_particles):
            schedule = np.zeros(self.n_families, dtype=int)
            for family_id in range(self.n_families):
                schedule[family_id] = np.random.choice(
                    np.arange(self.n_days),
                    p=choice_weights[family_id]
                ) + 1

            daily_occupancy = np.zeros(self.n_days + 1, dtype=int)
            np.add.at(daily_occupancy, schedule, self.family_size)
            
            if np.any((daily_occupancy[1:] < self.min_people) | (daily_occupancy[1:] > self.max_people)):
                schedule = self._repair_schedule_fast(schedule, daily_occupancy)
            
            particles[p] = schedule
        
        return particles
    
    def _repair_schedule_fast(self, schedule, daily_occupancy):
        """Faster version of schedule repair using vectorized operations where possible."""
        underflow_days = np.where(daily_occupancy[1:] < self.min_people)[0] + 1
        overflow_days = np.where(daily_occupancy[1:] > self.max_people)[0] + 1
        
        for day in underflow_days:
            needed_people = self.min_people - daily_occupancy[day]
            source_days = np.where(daily_occupancy[1:] > self.min_people + 20)[0] + 1
            if len(source_days) == 0:
                continue
                
            for source_day in source_days:
                potential_families = np.where(
                    (schedule == source_day) & 
                    (self.family_size <= needed_people)
                )[0]
                
                if len(potential_families) == 0:
                    continue
                
                family_sizes = self.family_size[potential_families]
                sort_idx = np.argsort(-family_sizes)  # Negative for descending order
                potential_families = potential_families[sort_idx]
                
                for family_id in potential_families:
                    family_size = self.family_size[family_id]
                    if (daily_occupancy[source_day] - family_size >= self.min_people and
                        daily_occupancy[day] + family_size <= self.max_people):
                        schedule[family_id] = day
                        daily_occupancy[source_day] -= family_size
                        daily_occupancy[day] += family_size
                        needed_people -= family_size
                        if needed_people <= 0:
                            break
                
                if needed_people <= 0:
                    break
        
        for day in overflow_days:
            excess_people = daily_occupancy[day] - self.max_people
            target_days = np.where(daily_occupancy[1:] < self.max_people - 20)[0] + 1
            if len(target_days) == 0:
                continue
            
            families_in_day = np.where(schedule == day)[0]
            if len(families_in_day) == 0:
                continue
            
            family_sizes = self.family_size[families_in_day]
            sort_idx = np.argsort(family_sizes)
            families_in_day = families_in_day[sort_idx]
            
            for family_id in families_in_day:
                family_size = self.family_size[family_id]
                target_days_sorted = target_days[np.argsort(daily_occupancy[target_days])]
                for target_day in target_days_sorted:
                    if (daily_occupancy[day] - family_size >= self.min_people and
                        daily_occupancy[target_day] + family_size <= self.max_people):
                        schedule[family_id] = target_day
                        daily_occupancy[day] -= family_size
                        daily_occupancy[target_day] += family_size
                        excess_people -= family_size
                        break
                if excess_people <= 0:
                    break
        
        return schedule

    def evaluate_fitness(self, schedule):
        total_cost = 0
        daily_occupancy = np.zeros(self.n_days + 1, dtype=int) 
        
        for family_id, assigned_day in enumerate(schedule):
            family_members = self.family_size[family_id]
            daily_occupancy[int(assigned_day)] += family_members  
            
            choice_idx = np.where(self.family_choices[family_id] == assigned_day)[0]
            if len(choice_idx) > 0:
                choice = choice_idx[0]
                total_cost += self.choice_costs[choice].calculate_cost(family_members)
            else:
                total_cost += self.choice_costs[-1].calculate_cost(family_members)
        
        for day in range(1, self.n_days + 1):
            if daily_occupancy[day] < self.min_people or daily_occupancy[day] > self.max_people:
                return float('inf')
        
        accounting_penalty = 0
        for d in range(self.n_days, 0, -1): 
            N_d = daily_occupancy[d]
            N_d_next = daily_occupancy[d] if d == self.n_days else daily_occupancy[d + 1]
            diff = N_d - 125
            if diff > 0:
                accounting_penalty += (diff / 400.0) * (N_d ** (0.5 + abs(N_d - N_d_next) / 50.0))
        
        total_cost += accounting_penalty
        
        return total_cost

    def update(self):
        self.w = max(0.4, self.w * 0.99) 
        r1, r2 = np.random.rand(2)
        self.velocities = (self.w * self.velocities + 
                          self.c1 * r1 * (self.p_best - self.particles) +
                          self.c2 * r2 * (self.g_best - self.particles))
        
        self.velocities = np.clip(self.velocities, -5, 5)
        
        new_positions = np.clip(np.round(self.particles + self.velocities), 1, self.n_days).astype(int)
        
        for i in range(self.n_particles):
            daily_occupancy = np.zeros(self.n_days + 1, dtype=int)
            for family_id, assigned_day in enumerate(new_positions[i]):
                daily_occupancy[assigned_day] += self.family_size[family_id]
            
            valid = True
            for day in range(1, self.n_days + 1):
                if daily_occupancy[day] < self.min_people or daily_occupancy[day] > self.max_people:
                    valid = False
                    break
            
            if not valid:
                new_positions[i] = self._repair_schedule_fast(new_positions[i], daily_occupancy)
            
            score = self.evaluate_fitness(new_positions[i])
            
            if score < self.p_best_scores[i] or (score < float('inf') and np.random.random() < 0.01):
                self.p_best[i] = new_positions[i].copy()
                self.p_best_scores[i] = score
                
                if score < self.g_best_score:
                    self.g_best = new_positions[i].copy()
                    self.g_best_score = score
        
        for i in range(self.n_particles):
            if np.random.random() < 0.1:  # 10% chance of mutation
                family_id = np.random.randint(0, self.n_families)
                if np.random.random() < 0.8:
                    new_positions[i, family_id] = np.random.choice(self.family_choices[family_id])
                else:
                    new_positions[i, family_id] = np.random.randint(1, self.n_days + 1)
        
        self.particles = new_positions


def solve_santa_workshop():
    n_particles = 100 
    n_families = 5000
    n_days = 100
    max_iterations = 2000 
    
    family_data = pd.read_csv('data/family_data.csv')
    
    families = []
    for _, row in family_data.iterrows():
        choices = [row[f'choice_{i}'] for i in range(10)]
        n_people = row['n_people']
        families.append(FamilyData(n_people, choices))
    
    pso = ParticleSwarmOptimizer(n_particles, n_families, n_days, families)
    
    best_score = float('inf')
    stagnation_counter = 0
    
    for iteration in range(max_iterations):
        pso.update()
        
        if pso.g_best_score < best_score:
            best_score = pso.g_best_score
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if stagnation_counter > 50:
            print(f"Reinitializing 50% of particles at iteration {iteration}")
            new_particles = pso.init_particles()
            replace_idx = np.random.choice(n_particles, n_particles // 2, replace=False)
            pso.particles[replace_idx] = new_particles[replace_idx]
            pso.velocities[replace_idx] = 0
            stagnation_counter = 0
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Score: {pso.g_best_score}")
    
    return pso.g_best, pso.g_best_score


best_schedule, best_score = solve_santa_workshop()
