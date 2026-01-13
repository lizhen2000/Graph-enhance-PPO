import torch
import random
import numpy as np
from collections import deque
from problem_set import *

def calculate_profit(arr: list):
    
    funvalue = 0
    f1 = 0
    f2 = 0
    f3 = 0

    for i in range(len(arr)):
        f1 += vp[0][arr[i]]
        f1 += tool[vp[6][arr[i]]][vp[5][arr[i]]]
        f2 += vp[1][arr[i]]
        f3 += cxcb[arr[i]]
    
    for i in range(len(arr) - 1):
        f1 += direction[vp[4][arr[i]]][vp[4][arr[i + 1]]]
        f1 += tool[vp[5][arr[i]]][vp[6][arr[i + 1]]]
    
    funvalue = f2 - f1 * 0.0111 - f3
    return funvalue

def calculate_carbonems(arr):
    
    funvalue = 0
    f4 = 0

    for i in range(len(arr)):
        f4 += dytpf[arr[i]]
    
    funvalue = f4
    return funvalue

def calculate_time(arr):
 
    funvalue = 0
    f1 = 0

    for i in range(len(arr)):
        f1 += vp[0][arr[i]]
        f1 += tool[vp[6][arr[i]]][vp[5][arr[i]]]
    
    for i in range(len(arr)-1):
        f1 += direction[vp[4][arr[i]]][vp[4][arr[i+1]]]
        f1 += tool[vp[5][arr[i]]][vp[6][arr[i+1]]]

    funvalue = f1
    return funvalue

class DynamicRewardScaler:
    def __init__(self, window_size=100, epsilon=1e-8):
        self.baseline = {
            'time': (1.0/2222, 1.0/2096),
            'profit': (11208.35, 11695.34),
            'carbon': (2835.06, 2938.42)
        }
        self.epsilon = epsilon


    def normalize(self, value, metric_type):

        min_val, max_val = self.baseline[metric_type]
        range_val = max_val - min_val + self.epsilon
        

        norm_val = (value - min_val) / range_val
        
        norm_val = torch.tensor(
            norm_val,
            dtype=torch.float32,
            device=device
        )

        return norm_val
    
def generate_preference_vector(mode = 'hybrid'):


    if mode == 'discrete':
        valid_combinations = []
        possible_values = [round(i * 0.01, 2) for i in range(10, 91)]

        for first in possible_values:
            for second in possible_values:
                third = round(1.0 - first - second, 2)
                if 0.1 <= third <= 0.9 and round(first + second + third, 2) == 1.0:
                    combination = [first, second, third]
                    valid_combinations.append(combination)

        selected_combination = random.choice(valid_combinations)
        np.random.shuffle(selected_combination)

        return torch.tensor(selected_combination, dtype=torch.float32, device=device)
    elif mode == 'continuous':
        weights = np.random.dirichlet(alpha=[1.0, 1.0, 1.0])
        return torch.tensor(weights, dtype=torch.float32, device=device)
    
    else:
        if np.random.random() < 0.90:
            return generate_preference_vector(mode='discrete')
        else:
            return generate_preference_vector(mode='continuous')

def generate_strategic_preference_vector(region_visits = None):

    grid_size = 7

    if region_visits is None or random.random() < 0.3:
        return generate_preference_vector('hybrid')
    
    all_regions = []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                z = (k + 0.5) / grid_size

                total = x + y + z
                if total > 0:
                    pref = [x/total, y/total, z/total]
                    region_id = (i, j, k)
                    visits = region_visits.get(region_id, 0)
                    all_regions.append((region_id, pref, visits))

    all_regions.sort(key=lambda x: x[2])

    candidate_count = max(1, int(len(all_regions) * 0.3))
    selected_region = random.choice(all_regions[:candidate_count])

    pref = selected_region[1]
    noise = [random.uniform(-0.05, 0.05) for _ in range(3)]
    noise[2] = -noise[0] - noise[1]
    pref = [max(0.05, min(0.95, p + n)) for p, n in zip(pref, noise)]
    
    total = sum(pref)
    pref = [p / total for p in pref]
    
    return torch.tensor(pref, dtype=torch.float32, device=device)

        
