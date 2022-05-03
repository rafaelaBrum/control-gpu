from gurobipy import Env, GRB, Model
from math import inf, ceil

# How do I divide by a variable in Gurobi?
# https://support.gurobi.com/hc/en-us/articles/360053259371-How-do-I-divide-by-a-variable-in-Gurobi-

# decision variable: x
# objective_function = (2 / x) + (10 * x)    ----> Gurobi does not support dividing by variables

# aux decision variable: z1
# transformed_objective_function = z1 + (10 * x)
# model.addConstr(2 == z1 * x, "c1")  ----> Constraint 'c1': z1 = 2 / x -> z1 * x = 2


def clean_model() -> None:
    with Env() as env, Model(name="Model Optimization on Gurobi for Python",
                             env=env) as model:
        # Set Model Parameters
        # gurobipy.GurobiError: Quadratic equality constraints are non-convex. Set NonConvex parameter to 2 to solve model.
        # https://www.gurobi.com/documentation/9.5/refman/nonconvex.html
        #model.setParam("NonConvex", 2)
        # How to make Gurobi not printing the optimization information?
        # https://support.gurobi.com/hc/en-us/community/posts/360065053272-How-to-make-Gurobi-not-printing-the-optimization-information-
        model.setParam("LogToConsole", 0)
        # Set Model Decision Variables
        # X
        x_lower_bound = 1
        x_upper_bound = inf
        x = model.addVar(name="x",
                         vtype=GRB.INTEGER,
                         lb=x_lower_bound,
                         ub=x_upper_bound,
                         obj=0,
                         column=None)
        # Y
        y_lower_bound = 1
        y_upper_bound = inf
        y = model.addVar(name="y",
                         vtype=GRB.INTEGER,
                         lb=y_lower_bound,
                         ub=y_upper_bound,
                         obj=0,
                         column=None)
        # Set Model Objective Function
        objective_function = 2 * x + 3 * y
        model.setObjective(objective_function, GRB.MINIMIZE)
        # Set Model Constraints
        model.addConstr(x <= 10, "constraint_1")
        model.addConstr(x + y >= 5, "constraint_2")
        # Optimize Model
        model.optimize()
        # If Model is Feasible (Found Optimal Value, GRB.OPTIMAL)...
        if model.status == 2:
            for v in model.getVars():
                if str(v.varName) == "x":
                    print("X = {0}".format(v.x))
                elif str(v.varName) == "y":
                    print("Y = {0}".format(v.x))
            print("Objective Function Value = {0}".format(objective_function.getValue()))
        del env
        del model


if __name__ == "__main__":
    clean_model()

