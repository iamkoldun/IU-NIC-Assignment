import numpy as np
from itertools import product
import time

# Constants
MIN_OCCUPANCY = 125
MAX_OCCUPANCY = 300
NUM_DAYS = 100
NUM_FAMILIES = 5000  # We'll use a smaller subset for testing

# Preference costs
PREFERENCE_COSTS = {
    0: 0,  # choice_0
    1: 50,  # choice_1
    2: 50 + 9,  # choice_2
    3: 100 + 9,  # choice_3
    4: 200 + 9,  # choice_4
    5: 200 + 18,  # choice_5
    6: 300 + 18,  # choice_6
    7: 300 + 36,  # choice_7
    8: 400 + 36,  # choice_8
    9: 500 + 36 + 199,  # choice_9
    10: 500 + 36 + 398  # otherwise
}

def calculate_accounting_cost(occupancy):
    """Calculate the accounting cost based on daily occupancy changes."""
    cost = 0
    for d in range(NUM_DAYS - 1):
        Nd = occupancy[d]
        Nd_plus_1 = occupancy[d + 1]
        cost += ((Nd - 125) / 400) * (Nd ** 0.5 + Nd_plus_1 ** 0.5)
    return cost

def calculate_preference_cost(assignments, family_preferences):
    """Calculate the total preference cost for all families."""
    total_cost = 0
    for family_idx, day in enumerate(assignments):
        preference = family_preferences[family_idx]
        if day == preference:
            cost = PREFERENCE_COSTS[0]
        else:
            cost = PREFERENCE_COSTS[10]  # Using the maximum cost for simplicity
        total_cost += cost
    return total_cost

def is_valid_schedule(assignments, family_sizes):
    """Check if the schedule meets daily occupancy constraints."""
    daily_occupancy = np.zeros(NUM_DAYS)
    for family_idx, day in enumerate(assignments):
        daily_occupancy[day] += family_sizes[family_idx]
    
    
    return np.all((daily_occupancy >= MIN_OCCUPANCY) & (daily_occupancy <= MAX_OCCUPANCY))

def brute_force_schedule(family_preferences, family_sizes, num_families=10):
    """Brute force approach to find optimal schedule."""
    best_cost = float('inf')
    best_schedule = None
    
    # Generate all possible combinations for the first 'num_families' families
    possible_days = range(NUM_DAYS)
    total_combinations = NUM_DAYS ** num_families
    print(f"Total combinations to check: {total_combinations}")
    
    for i, schedule in enumerate(product(possible_days, repeat=num_families)):
        if i % 100000 == 0:  # Print progress every 100,000 combinations
            print(f"Checking combination {i}/{total_combinations}")
        
        if is_valid_schedule(schedule, family_sizes[:num_families]):
            # Calculate costs
            pref_cost = calculate_preference_cost(schedule, family_preferences[:num_families])
            daily_occupancy = np.zeros(NUM_DAYS)
            for family_idx, day in enumerate(schedule):
                daily_occupancy[day] += family_sizes[family_idx]
            acc_cost = calculate_accounting_cost(daily_occupancy)
            
            total_cost = pref_cost + acc_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_schedule = schedule
                print(f"Found better schedule with cost {best_cost}: {best_schedule}")
    
    return best_schedule, best_cost

def main():
    # Generate sample data
    np.random.seed(42)
    family_preferences = np.random.randint(0, NUM_DAYS, size=NUM_FAMILIES)
    family_sizes = np.random.randint(1, 6, size=NUM_FAMILIES)  # Assuming 1-5 family members
    
    # Run with a small subset of families
    num_test_families = 50  # Increased number of families to make it possible to meet occupancy requirements
    print(f"Running brute force with {num_test_families} families...")
    print(f"Family sizes for test: {family_sizes[:num_test_families]}")
    print(f"Total family members: {sum(family_sizes[:num_test_families])}")
    print(f"Family preferences for test: {family_preferences[:num_test_families]}")
    
    # Calculate average family size
    avg_family_size = np.mean(family_sizes[:num_test_families])
    print(f"Average family size: {avg_family_size:.2f}")
    
    # Calculate minimum number of families needed to meet occupancy
    min_families_needed = MIN_OCCUPANCY / avg_family_size
    print(f"Minimum families needed to meet occupancy: {min_families_needed:.2f}")
    
    start_time = time.time()
    best_schedule, best_cost = brute_force_schedule(family_preferences, family_sizes, num_test_families)
    end_time = time.time()
    
    print(f"Best schedule found: {best_schedule}")
    print(f"Best cost: {best_cost}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
