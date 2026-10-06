import numpy as np
from sympy import Eq, Equality, diff, Function, Symbol, lambdify, Rational
from sympy.solvers import dsolve
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
- The spread rate is derived from the product of the initial population
and a user-supplied proportionality constant (k): r = k*P_0
- The proportionality constant (k) is intended to satisfy the relationship
0 < k <= 0.05 to prevent explosion or inflated answers
- All rabbits are equally capable of receiving and transmitting alerts.
- Environmental factors (terrain, vegetation, weather, visibility)
  are ignored.
- Predator behavior is not modeled.
- Once a rabbit becomes alerted, it remains alerted for the duration
  of the simulation.
- Time is treated as continuous in the differential equation model.
- Population size is fixed and does not account for rabbits born or dead

Limitations:
- The model does not account for communication failures.
- The model does not account for spatial distance between rabbits.
- The model does not distinguish between adults, juveniles, or rabbits
  with offspring.
- The proportionality constant is hypothetical and should not be interpreted
  as a measured biological quantity or a realistic representation of rabbit
  behavior.

Future Work:
- Estimate spread-rate parameters from observational data.
- Introduce spatial constraints on rabbit communication.
- Add stochastic (probabilistic) signaling behavior.
- Integrate dynamic population growth and death into the model.
"""

def form_differential_equation(spread_rate: float, initial_population: int):
    A = Function('A')
    return Eq(diff(A(t), t), spread_rate * initial_population * A(t) * (1 - A(t) / initial_population))

def solve_equation(equation: Eq, a_0: int):
    A = Function('A')
    solution = dsolve(equation, ics={A(0): a_0}, hint='Bernoulli')
    return solution

# deprecated: new approach is continuous instead of discrete and considers the whole function
def gather_data(solution: Equality):
    right = solution.rhs
    times = []
    num_alerted = []
    for i in range(5):
        times.append(i)
        num_alerted.append(right.subs(t, i))
    return times, num_alerted

def gather_data(solution: Equality,
                start_time: float,
                end_time: float,
                num_points: int):
    time_values = np.linspace(start_time, end_time, num_points)

    alert_function = lambdify(
        t,
        solution.rhs,
        modules=['numpy']
    )

    alert_values = alert_function(time_values)

    return time_values, alert_values

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Insert parameters for modeling rabbit alert calls.")
    parser.add_argument("--capacity", type=int, default=15, help="Maximum number of rabbits in the population")
    parser.add_argument("--initial_alerted", type=int, default=1, help="Number of rabbits that detect the predator immediately")
    parser.add_argument("--spread_rate", type=float, default = 0.1, help="Fastness of how rabbits alert each other")
    args = parser.parse_args()
    
    rate = Rational(str(args.spread_rate))
    equation = form_differential_equation(rate, args.capacity)
    solution = solve_equation(equation, args.initial_alerted)
    
    time_values = np.linspace(0, 5, 100)
    alert = lambdify(t, solution.rhs, modules=['numpy'])
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
    
    plt.savefig(f"rabbit_alerts_naive_initial_{args.initial_alerted}_rate_{args.spread_rate}_logistic.png")