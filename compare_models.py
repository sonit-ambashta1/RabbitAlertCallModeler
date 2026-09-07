import numpy as np

from exponential_model import form_differential_equation, solve_equation, gather_data
import matplotlib.pyplot as plt
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Insert parameters for modeling rabbit alert calls.")
    parser.add_argument("--initial_alerted_points", type=int, nargs="+", default=[1, 2, 3], help="Number of rabbits that detect the predator immediately")
    parser.add_argument("--spread_rates", type=float, nargs="+", default = np.arange(0.05, 1.0, 0.05), help="Fastness of how rabbits alert each other")
    args = parser.parse_args()
    
    for initial_point in args.initial_alerted_points:
        plt.figure(figsize=(10,6))
        plt.xlabel("Time (steps)")
        plt.ylabel("Number of alerted rabbits")
        if initial_point == 1:
            plt.title(f"Rabbit Alert Call Model Distribution with 1 Initial Alerted Rabbit")
        else:
            plt.title(f"Rabbit Alert Call Model Distribution with {initial_point} Initial Alerted Rabbits")
        for rate in args.spread_rates:
            equation = form_differential_equation(rate)
            solution = solve_equation(equation, initial_point)
            x, y = gather_data(solution)
            plt.plot(x, y, label=f"Growth Rate: {rate}")
        plt.legend()
        plt.savefig(f"rabbit_alerts_{initial_point}_spread_rates.png")
        plt.close()