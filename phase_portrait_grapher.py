"""
Assumptions:
since the rabbit population and time will have to be positive, only the first quadrant of the slope field will be graphed.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, Piecewise, solve, lambdify, diff

def graph_phase_line(equation: Eq, t_range=(0, 5), y_range=(-15, 15), density=20):
    # approach:
    # 1. Plot equilibrium lines vertical markers
    t = symbols('t')
    A = Function('A')(t)
    autonomous_de = equation.subs({diff(A, t): 0})
    results = solve(autonomous_de, A)
    print(results)
    
    # cast to a set to remove duplicates (choose the midpoint between duplicates)
    results = list(set(results))
    results = sorted(results)
    results.insert(0, y_range[0])
    results.append(y_range[1])
    print(results)

    plt.figure(figsize=(10, 6))
    plt.xlim(t_range[0], t_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.title("Phase Portrait of Rabbit Alert Call Model")
    plt.xlabel("Time (t)")
    plt.ylabel("Number of Alerted Rabbits (A(t))")
        
    # it is assumed that the results are equilibrium points
    for i in range(1, len(results)):
        # logic to plot the equilibrium lines
        if i != len(results) - 1:
            plt.axhline(y=results[i], color='black', linestyle='--', zorder=2)
        
        # shade and label regions between equilibrium lines
        midpoint = (results[i] + results[i-1]) / 2
        slope = determine_slope(equation, midpoint)
        color = 'green' if slope > 0 else 'red'
        plt.axhspan(results[i-1], results[i], facecolor=color, alpha=0.3, zorder=1)
        plt.text(t_range[0], midpoint, f'Slope: {slope:.2f}', verticalalignment='center', zorder=3)
        
    plt.legend()
    plt.savefig("phase_portrait.png")

# pick a random autonomous equation to determine increasing or decreasing
def determine_slope(equation: Eq, A_value: float):
    t = symbols('t')
    A = Function('A')(t)
    slope = equation.subs({A: A_value})
    print(slope.rhs)
    return slope.rhs.evalf()

if __name__ == "__main__":
    equation = Eq(diff(Function('A')(symbols('t')), symbols('t')), 0.1 * Function('A')(symbols('t')) * (1 - Function('A')(symbols('t')) / 15))
    graph_phase_line(equation, y_range=(-25, 25), density=20)
    print(determine_slope(equation, 5))  # Example usage to determine slope at A=5