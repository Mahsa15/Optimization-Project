from pyscipopt import Model, quicksum
import numpy as np
import random
import logging
import sys

logging.basicConfig(filename='ilp_process.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')


def setup_problem(file_path):
    model = Model("ILP from LP File")
    try:
        model.readProblem(file_path)
        logging.info(f'Model loaded successfully from {file_path}')
    except Exception as e:
        logging.error(f'Failed to load model from {file_path}: {e}')
    return model


def vu_values(n):
    v = [[np.cos(2 * np.pi * i * j / n) for i in range(n)] for j in range((n - 1) // 2 + 1)]
    u = [[np.sin(2 * np.pi * i * j / n) for i in range(n)] for j in range((n - 1) // 2 + 1)]
    logging.info(f'Generated v and u values for {n} variables: v={v}, u={u}')
    return v, u


def create_pv_pu(model, v, u, x, j):
    p_v = quicksum(v[j][i] * x[i] for i in range(len(x)))
    p_u = quicksum(u[j][i] * x[i] for i in range(len(x)))
    logging.info(f'Projections expressions created for j={j}')
    return p_v, p_u



def creating_P0(model, v, u):
    n = len(model.getVars())
    x = model.getVars()
    M = 10000
    r = [model.addVar(vtype="BINARY", name=f"r_{j}") for j in range((n-1)//2 + 1)]

    for j in range(((n-1) // 2) + 1):
        p_v, p_u = create_pv_pu(model, v, u, x, j)

        if j == 0 or (n % 2 == 0 and j == n // 2):
            model.addCons(p_v <= r[j] * M)
            model.addCons(p_v >= -r[j] * M)
        else:
            p_j = (p_v * p_v + p_u * p_u)
            model.addCons(p_j <= r[j] * M)

        logging.debug(f'Added constraints for variable set {j}')

    model.addCons(quicksum(r[j] for j in range(((n-1) // 2) + 1)) <= ((n) // 2))
    logging.info('All P0 constraints added')

    output_lp_path = "1-P_0-const.lp"
    model.writeProblem(output_lp_path)
    logging.info(f'P0 model written to {output_lp_path}')
    print(f"Model with new constraints written to: {output_lp_path}")

    model.optimize()
    if model.getStatus() == "optimal":
        logging.info(f'P0 solved optimally with objective: {model.getObjVal()}')
        return model.getObjVal()
    else:
        logging.warning('P0 optimization is infeasible')
        return -np.inf

def solve_lp_relaxation(model):
    for var in model.getVars():
        if var.vtype() in ["INTEGER", "BINARY"]:
            model.chgVarType(var, 'C')
    model.optimize()
    if model.getStatus() == "optimal":
        logging.info(f'LP relaxation solved with objective: {model.getObjVal()}')
        return model.getObjVal()
    else:
        logging.warning(f'LP relaxation is infeasible. Status: {model.getStatus()}')
        return -np.inf

def create_essential_and_projected_sets(k, n):
    base_value = k // n
    remainder = k % n
    ucp = [base_value] * n
    for _ in range(remainder):
        index = random.choice(range(n))
        ucp[index] += 1
    atom_points = []
    #for _ in range(2):  
    #    i, j = random.sample(range(n), 2)
    #    new_point = ucp.copy()
    #    new_point[i] += 1
    #    new_point[j] -= 1
    #    atom_points.append(new_point)
    essential_set = [ucp] + atom_points
    projected_set = [[x - base_value for x in point] for point in essential_set]
    logging.info(f'Essential set for layer {k}: {essential_set}')
    logging.info(f'Projected set for layer {k}: {projected_set}')
    return essential_set, projected_set
    

def check_feasibility_of_point(model, point):
    model.freeTransform()
    for var, value in zip(model.getVars(), point):
        model.chgVarLb(var, value)
        model.chgVarUb(var, value)
    model.writeProblem("1-feasibility_check.lp")
    model.optimize()
    if model.getStatus() == "optimal" and model.getObjVal() == sum(point):
        logging.info(f"Feasible solution found: {point}") 
        print("Feasible solution found:", point)
        return True
    else:
        logging.info(f"Checked point {point} is not feasible. Status: {model.getStatus()}")
        return False

def feasibility_check_by_layer(model, n, l, l_prime):
    for layer in range(l, l_prime - 1, -1):
        essential_set, _ = create_essential_and_projected_sets(layer, n)
        logging.info(f'Checking feasibility for layer {layer}')
        print(f"Checking feasibility for layer {layer}")
        for point in essential_set:
            if check_feasibility_of_point(model, point):
                print("Solution found at layer", layer, "- stopping further search.")
                logging.info(f'Solution found at layer {layer} with point {point}')
                return True
        print(f"No feasible solutions found for layer {layer}.")
        logging.info(f'No feasible solutions found for layer {layer}')
    return False

def compute_T_k(n, x, k, model):
    T_k_value = 0
    for m in range(1, (n-1)//2 + 1):
        cos_m = [np.cos(2 * np.pi * m * i / n) for i in range(n)]
        sin_m = [np.sin(2 * np.pi * m * i / n) for i in range(n)]
        
        p_v = quicksum(cos_m[i] * x[i] for i in range(n))
        p_u = quicksum(sin_m[i] * x[i] for i in range(n))
        p_j_squared = p_v * p_v + p_u * p_u
        
        model.addCons(p_j_squared >= 1e-6)
        
        shifted_cos_m = np.roll(cos_m, -k)
        p_sigma = quicksum(shifted_cos_m[i] * x[i] for i in range(n))
        
        T_k_value += 2 * p_sigma / p_j_squared 

    logging.info(f'Computed T_k value for k={k}: {T_k_value}')
    return T_k_value

def add_constraints_and_optimize(model, x, n, projected_set, layer):
    model.freeTransform()
    constraints_added = []
    for point in projected_set:
        T_constraints = quicksum(point[k] * compute_T_k(n, x, k, model) for k in range(n))
        cons = model.addCons(T_constraints <= 0)
        constraints_added.append(cons)
    
    
    model_name = f"model_new_constraints_layer_{layer}.cip"
    model.writeProblem(model_name)
    logging.info(f'Model with new constraints written to {model_name}')
    
    model.optimize()
    if model.getStatus() == "optimal":
        print(f"Optimal value for layer {layer}:", model.getObjVal())
        print("Solution:", [model.getVal(var) for var in x])
        logging.info(f'Optimal value for layer {layer}: {model.getObjVal()} and the solution: {[model.getVal(var) for var in x]}')
        return True
    else:
        print(f"No optimal solution found for layer {layer}.")
        logging.info(f'No optimal solution found for layer {layer}. Status: {model.getStatus()}')
        return False 
def main():
    
    if len(sys.argv) < 2:
        logging.error("No LP file path provided.")
        return
    
    file_path = sys.argv[1]
    
    # Create model for P0
    model_for_P0 = setup_problem(file_path)
    n = len(model_for_P0.getVars())
    v, u = vu_values(n)
    f1 = creating_P0(model_for_P0, v, u)
    print(f"Objective value from P0: {f1}")
    logging.info(f'Objective value from P0: {f1}')
    
    model_for_LP = setup_problem(file_path)
    l = solve_lp_relaxation(model_for_LP)
    logging.info(f'LP relaxation value: {l}')
    print(f"LP relaxation value: {l}")
    if l == -np.inf:
        print("LP relaxation infeasible or failed to solve.")
        logging.warning("LP relaxation infeasible.")
        return max(f1, l)

    l_prime = (int(l) // n) * n

    # loop through layers
    for layer in range(int(l), l_prime - 1, -1):
        model_for_layer = setup_problem(file_path)  
        essential_set, projected_set = create_essential_and_projected_sets(layer, n)
        logging.info(f'Processing layer {layer}')
        
        feasible_found = False
        for point in essential_set:
            if check_feasibility_of_point(model_for_layer, point):
                print(f"Feasible solution found at layer {layer} with point {point}.")
                logging.info(f'Feasible solution found at layer {layer} with point {point}')
                feasible_found = True
                break

        if feasible_found:
            final_value = max(f1, model_for_layer.getObjVal())
            logging.info(f'Solution found, final value: {final_value}')
            return final_value
        
        model_for_constraints = setup_problem(file_path)  
        x = model_for_constraints.getVars()  
        if add_constraints_and_optimize(model_for_constraints, x, n, projected_set, layer):
            print(f"Optimal value after adding new constraints at layer {layer}: {model_for_constraints.getObjVal()}")
            final_value = max(f1, model_for_constraints.getObjVal())
            logging.info(f'Optimal value after constraints added: {final_value}')
            return final_value

    print("No feasible solution found in any layer.")
    logging.info("No feasible solution found in any layer.")
    final_value = max(f1, -np.inf)
    logging.info(f'Final result: {final_value}')
    return final_value

if __name__ == "__main__":
    final_result = main()
    print(f"Final result: {final_result}")
