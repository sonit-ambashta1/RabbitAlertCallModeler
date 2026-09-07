import numpy as np

import exponential_model
import capped_exponential_model
import matplotlib.pyplot as plt
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Insert parameters for modeling rabbit alert calls.")
    parser.add_argument("--initial_alerted_points", type=int, nargs="+", default=[1, 2, 3], help="Number of rabbits that detect the predator immediately")
    parser.add_argument("--spread_rates", type=float, nargs="+", default = np.arange(0.05, 1.0, 0.05), help="Fastness of how rabbits alert each other")
    parser.add_argument("--capacity", type=int, default = 15, help="Number of rabbits as part of the habitat")
    args = parser.parse_args()
    
    for initial_point in args.initial_alerted_points:
        # plot naive exponential model with combinations of initial conditions and spread rates
        plt.figure(figsize=(10,6))
        plt.xlabel("Time (steps)")
        plt.ylabel("Number of alerted rabbits")
        if initial_point == 1:
            plt.title(f"Rabbit Alert Call Model Distribution with 1 Initial Alerted Rabbit")
        else:
            plt.title(f"Rabbit Alert Call Model Distribution with {initial_point} Initial Alerted Rabbits")
        for rate in args.spread_rates:
            equation = exponential_model.form_differential_equation(rate)
            solution = exponential_model.solve_equation(equation, initial_point)
            x, y = exponential_model.gather_data(solution, start_time=0, end_time=5, num_points=100)
            plt.plot(x, y, label=f"Growth Rate: {round(rate, 2)}")
        plt.legend()
        plt.savefig(f"rabbit_alerts_{initial_point}_spread_rates_naive_exponential.png")
        plt.close()
        
        # plot capped exponential model with combinations of initial conditions and spread rates
        plt.figure(figsize=(10,6))
        plt.xlabel("Time (steps)")
        plt.ylabel("Number of alerted rabbits")
        if initial_point == 1:
            plt.title(f"Rabbit Alert Call Model Distribution with 1 Initial Alerted Rabbit (Capped Exponential)")
        else:
            plt.title(f"Rabbit Alert Call Model Distribution with {initial_point} Initial Alerted Rabbits (Capped Exponential)")
        for rate in args.spread_rates:
            equation = capped_exponential_model.form_differential_equation(rate)
            solution = capped_exponential_model.solve_equation(equation, initial_point)
            poi = capped_exponential_model.find_point_of_intersection(solution, args.capacity)
            piecewise_solution = capped_exponential_model.form_piecewise_solution(solution, args.capacity, poi)
            x, y = capped_exponential_model.gather_data(piecewise_solution, start_time=0, end_time=5, num_points=100)
            
            plt.plot(x, y, label=f"Growth Rate: {round(rate, 2)}")
        plt.legend()
        plt.savefig(f"rabbit_alerts_{initial_point}_spread_rates_capped_exponential.png")
        plt.close()
