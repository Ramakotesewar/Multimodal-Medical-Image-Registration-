import numpy as np
import time


# Gorilla Troops Optimizer (GTO)

def boundaryCheck(Positions, lb, ub):
    for i in range(Positions.shape[0]):
        Flag4ub = Positions[i, :] > ub
        Flag4lb = Positions[i, :] < lb
        Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
    return Positions


def PROPOSED(X, fobj, lower_bound, upper_bound, max_iter):
    pop_size, variables_no = X.shape[0], X.shape[1]
    ub = upper_bound[0, :]
    lb = lower_bound[0, :]
    # initialize Silverback
    Silverback = []
    Silverback_Score = float('inf')
    convergence_curve = np.zeros((max_iter, 1))
    Pop_Fit = np.zeros((pop_size))
    for i in np.arange(1, pop_size + 1).reshape(-1):
        Pop_Fit[i] = fobj(X[i, :])
        if Pop_Fit[i] < Silverback_Score:
            Silverback_Score = Pop_Fit[i]
            Silverback = X[i, :]
    GX = X
    ##  Controlling parameter
    p = 0.03
    Beta = 3
    w = 0.8
    It = 0
    ct = time.time()
    ##Main loop
    for It in np.arange(1, max_iter + 1).reshape(-1):
        a = (np.cos(2 * np.random.rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * np.random.rand() - 1)
        ## Exploration:
        for i in np.arange(1, pop_size + 1).reshape(-1):
            if np.random.rand() < p:
                GX[i, :] = (ub - lb) * np.random.rand() + lb
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(- a, a, variables_no)
                    H = np.multiply(Z, X[i, :])
                    GX[i, :] = (np.random.rand() - a) * X[np.random.randint(np.array([1, pop_size])), :] + np.multiply(
                        C, H)
                else:
                    # update using GRO
                    coworkers = np.random.choice(np.delete(np.arange(pop_size), i), 2, replace=False)
                    digger1, digger2 = coworkers
                    for d in range(variables_no):
                        r1 = np.random.rand()
                        D3 = GX[digger2, d] - GX[digger1, d]
                        GX[i, d] = GX[i, d] + r1 * D3
        GX = boundaryCheck(GX, lower_bound, upper_bound)
        # Group formation operation
        for i in np.arange(1, pop_size + 1).reshape(-1):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
        ## Exploitation:
        for i in np.arange(1, pop_size + 1).reshape(-1):
            if a >= w:
                g = 2 ** C
                delta = (np.abs(np.mean(GX)) ** g) ** (1 / g)
                GX[i, :] = np.multiply(C * delta, (X[i, :] - Silverback)) + X[i, :]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.randn(1, variables_no)
                else:
                    h = np.random.randn(1, 1)
                r1 = np.random.rand()
                GX[i, :] = Silverback - np.multiply(
                    (Silverback * (2 * r1 - 1) - X[i, :] * (2 * r1 - 1)), (Beta * h))
        GX = boundaryCheck(GX, lower_bound, upper_bound)
        # Group formation operation
        for i in np.arange(1, pop_size + 1).reshape(-1):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :]
        convergence_curve[It] = Silverback_Score
    ct = time.time() - ct
    return Silverback_Score, convergence_curve, Silverback, ct
