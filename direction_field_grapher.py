from sympy import symbols, Function, dsolve, Derivative, Eq
import matplotlib.pyplot as plt
import numpy as np
import argparse

def form_population_equation(initial_population: int, births: float, deaths: float):
    """Form a differential equation for the population based on birth and death rates."""
    P = Function('P')
    t = symbols('t')
    equation = Eq(Derivative(P(t), t), births - deaths)
    solution = dsolve(equation, ics={P(0): initial_population})
    return solution

def form_logistic_differential_equation(prop_const: float, carrying_capacity: int):
    """Form the logistic differential equation based on the given parameters."""
    y = Function('y')
    t = symbols('t')
    return Eq(Derivative(y(t), t), prop_const * y(t) * (1 - y(t) / carrying_capacity))

def form_logistic_variation_differential_equation(prop_const: float, population_equation: Eq):
    """Form a logistic differential equation but the population is non-zero or variable"""
    y = Function('y')
    t = symbols('t')
    print(population_equation.rhs)
    return Eq(Derivative(y(t), t), prop_const * y(t) * (1 - y(t) / population_equation.rhs))

def slope_to_vector(slope):
    """ convert a slope to a unit vector for plotting"""
    return np.array([1, slope]) / np.sqrt(1 + slope**2)


if __name__ == "__main__":
    # the main file takes in logistic differential equations but will support multiple models
    parser = argparse.ArgumentParser(description="Plot the direction field for the logistic differential equation")
    parser.add_argument("--t_start", type=float, default=0, help="Start time for the direction field plot")
    parser.add_argument("--t_end", type=float, default=20, help="End time for the direction field plot")
    parser.add_argument("--y_start", type=float, default=-25, help="Start value for the y-axis in the direction field plot")
    parser.add_argument("--y_end", type=float, default=25, help="End value for the y-axis in the direction field plot")
    parser.add_argument("--prop_const", type=float, default=0.005, help="Proportionality constant for the logistic differential equation")
    parser.add_argument("--carrying_capacity", type=int, default=100, help="Carrying capacity for the logistic differential equation")
    args = parser.parse_args()
    logistic_de = form_logistic_differential_equation(args.prop_const, args.carrying_capacity)
    population_equation = form_population_equation(args.carrying_capacity, births=5, deaths=3)
    print(population_equation)
    logistic_var_de = form_logistic_variation_differential_equation(args.prop_const, population_equation)
    print(logistic_var_de)

    # plot the direction field for the logistic differential equation
    plt.figure(figsize=(10, 6))
    t_vals = np.arange(args.t_start, args.t_end + 1, 1)
    y_vals = np.arange(args.y_start, args.y_end + 1, 1)

    plt.xlim(args.t_start, args.t_end)
    plt.ylim(args.y_start, args.y_end)

    X, Y = np.meshgrid(t_vals, y_vals)
    print(X)
    print(Y)

    # S = args.prop_const * Y * (1 - Y / args.carrying_capacity)
    S = args.prop_const * Y * (1 - Y / population_equation.rhs)  # use the population equation for the logistic variation
    U = np.ones_like(S)
    V = S

    norm = np.sqrt(U**2 + V**2)
    U /= norm
    V /= norm

    plt.quiver(X, Y, U, V, color='blue', alpha=0.5)
    plt.savefig(f"logistic_direction_field_prop_const_{args.prop_const}_carrying_capacity_{args.carrying_capacity}.png")