"""
Python script to write jobs
"""

import numpy as np
import itertools

q_values = np.linspace(0, 0.60, 60)
p_values = (0, 0.05)
alpha_values = np.linspace(1.01, 1.50, 49)

for p, q, alpha in itertools.product(p_values, q_values, alpha_values):

    print(
        f"uv run main.py --repetitions 50 --years 200 --initial_population_size 10000 --probability_of_trauma {p:.02f} --probability_of_heal {q:.02f} --alpha {alpha:.02f}"
    )
