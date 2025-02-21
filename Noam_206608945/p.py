import math
from scipy.optimize import fsolve

def f(x):
    return 3 * math.sin(10 * x) - math.cos(x / 5) * x + x + 2

# Initial guess
initial_guess = 0

# Solve for the root
root = fsolve(f, initial_guess)[0]

# Print root and test the function value
print(f"Root: {root}")
print(f"f({root}) = {f(root)}")
