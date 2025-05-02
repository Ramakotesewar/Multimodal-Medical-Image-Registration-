import numpy as np
import time


##  Walrus Optimization Algorithm (WOA)
def WOA(positions, objective_function, LB, UB, Max_iterations):
    # Lower and Upper limit for variables
    lb = LB[0, :]
    ub = UB[0, :]
    SearchAgents, dimension = positions.shape
    Xbest = np.zeros(dimension)
    best_fitness = np.zeros(SearchAgents)
    for n in range(SearchAgents):
        best_fitness[n] = objective_function(positions[n, :])
    Convergence_curve = np.zeros((1, Max_iterations))
    ct = time.time()
    # Main optimization loop
    for t in range(Max_iterations):
        # Phase 1: Feeding strategy (Exploration)
        SW = Xbest  # Strongest walrus with the best value for the objective function
        for i in range(SearchAgents):
            I = np.round(1 + np.random.rand())
            X_P1 = positions[i, :] + np.random.rand(dimension) * (SW - I * positions[i, :])
            X_P1 = np.maximum(X_P1, lb)
            X_P1 = np.minimum(X_P1, ub)

            L = X_P1
            F_P1 = objective_function(L)
            if F_P1 < best_fitness[i]:
                positions[i, :] = X_P1
                best_fitness[i] = F_P1

        # Phase 2: Migration
        for i in range(SearchAgents):
            I = np.round(1 + np.random.rand())
            K = np.random.permutation(SearchAgents)
            K = K[K != i]
            X_K, F_RAND = positions[K[0], :], best_fitness[K[0]]
            if best_fitness[i] > F_RAND:
                X_P2 = positions[i, :] + np.random.rand() * (X_K - I * positions[i, :])
            else:
                X_P2 = positions[i, :] + np.random.rand() * (positions[i, :] - X_K)

            X_P2 = np.maximum(X_P2, lb)
            X_P2 = np.minimum(X_P2, ub)

            L = X_P2
            F_P2 = objective_function(L)
            if F_P2 < best_fitness[i]:
                positions[i, :] = X_P2
                best_fitness[i] = F_P2

        # Phase 3: Escaping and fighting against predators (Exploitation)
        LO_LOCAL = lb / t
        HI_LOCAL = ub / t
        for i in range(SearchAgents):
            I = np.round(1 + np.random.rand())
            X_P3 = positions[i, :] + LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL)
            X_P3 = np.maximum(X_P3, LO_LOCAL)
            X_P3 = np.minimum(X_P3, HI_LOCAL)
            X_P3 = np.maximum(X_P3, lb)
            X_P3 = np.minimum(X_P3, ub)

            L = X_P3
            F_P3 = objective_function(L)
            if F_P3 < best_fitness[i]:
                positions[i, :] = X_P3
                best_fitness[i] = F_P3

        # Update the best candidate solution
        best, location = np.min(best_fitness), np.argmin(best_fitness)
        if best < best_fitness:
            best_fitness = best
            Xbest = positions[location, :]
        Convergence_curve[:, t] = np.min(best_fitness)
    ct -= time.time()
    return best_fitness, Convergence_curve, Xbest, ct
