import numpy as np
from sympy import Eq, Equality, diff, Function, Symbol, lambdify, Piecewise
from sympy.solvers import dsolve, solve
from sympy.abc import t
import matplotlib.pyplot as plt
import argparse

"""
Rabbit Threat Detection Differential Equation Modeler

Models the spread of predator awareness throughout a rabbit population
using differential equations.

Motivation:
Inspired by research suggesting that European rabbits may use visual
signals (e.g., tail-flagging) to communicate danger to conspecifics.

Assumptions:
- The rabbit population contains at least one rabbit.
- One or more rabbits may initially detect a predator.
- The spread rate is a user-supplied parameter and is not currently
  estimated from observed rabbit behavior.
- The spread rate is intended to satisfy the relationship 0 < r <= 1 to prevent explosion or inflated answers
- All rabbits are equally capable of receiving and transmitting alerts.
- Environmental factors (terrain, vegetation, weather, visibility)
  are ignored.
- Predator behavior is not modeled.
- Once a rabbit becomes alerted, it remains alerted for the duration
  of the simulation.
- Time is treated as continuous in the differential equation model.

Limitations:
- The model does not account for communication failures.
- The model does not account for spatial distance between rabbits.
- The model does not distinguish between adults, juveniles, or rabbits
  with offspring.
- The spread-rate parameter is hypothetical and should not be interpreted
  as a measured biological quantity.

Future Work:
- Estimate spread-rate parameters from observational data.
- Introduce spatial constraints on rabbit communication.
- Add stochastic (probabilistic) signaling behavior.
- Compare exponential, logistic, and capped-growth models.
"""

def form_differential_equation(spread_rate: float):
    A = Function('A')
    return Eq(diff(A(t), t), spread_rate * A(t))

def solve_equation(equation: Eq, a_0: int):
    A = Function('A')
    solution = dsolve(equation, ics={A(0): a_0})
    return solution

# deprecated: new approach is continuous instead of discrete and considers the whole function
def gather_data(solution: Equality, carrying_capacity: int):
    right = solution.rhs
    times = []
    num_alerted = []
    for i in range(5):
        times.append(i)
        num_alerted.append(min(right.subs(t, i), carrying_capacity))
    return times, num_alerted

def find_point_of_intersection(solution: Equality, carrying_capacity: int):
    equation = Eq(carrying_capacity, solution.rhs)
    point = solve(equation, t)
    return point[0]

def form_piecewise_solution(solution: Equality, carrying_capacity: int, poi: float):
    piecewise_func = Piecewise(
        (solution.rhs, t < poi),
        (carrying_capacity, t >= poi)
    )
    return piecewise_func

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Insert parameters for modeling rabbit alert calls.")
    parser.add_argument("--initial_alerted", type=int, default=1, help="Number of rabbits that detect the predator immediately")
    parser.add_argument("--spread_rate", type=float, default = 0.1, help="Fastness of how rabbits alert each other")
    parser.add_argument("--capacity", type=int, default = 15, help="Number of rabbits as part of the habitat")

    args = parser.parse_args()
    
    equation = form_differential_equation(args.spread_rate)
    solution = solve_equation(equation, args.initial_alerted)
    
    find_cap_point = find_point_of_intersection(solution, args.capacity)
    piecewise_solution = form_piecewise_solution(solution, args.capacity, find_cap_point)
    
    time_values = np.linspace(0, 5, 100)
    alert = lambdify(t, piecewise_solution, modules=['numpy'])
    alert_values = alert(time_values)
    
    print(f"EQUATION: {equation}")
    print(f"SOLUTION: {solution}")
    
    print("DATA GATHERING:")
    print(f"t: {time_values}")
    print(f"A(t): {alert_values}")
    
    # prepare the plot
    plt.figure(figsize=(10,6))
    plt.title("Rabbit Alert Calls")
    plt.xlabel("Time")
    plt.ylabel("Number of Alerted Rabbits")
    plt.plot(time_values, alert_values)
    plt.axvline(x=find_cap_point, color="red")
    
    plt.savefig(f"rabbit_alerts_naive_initial_{args.initial_alerted}_rate_{args.spread_rate}_capped.png")