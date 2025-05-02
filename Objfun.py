import numpy as np
from Evaluate_Error import evaluat_error
from Global_Vars import Global_Vars
from Model_ATRUNet import Model_ATRUNet


def Objective_Function(Soln):
    CT = Global_Vars.CT
    MR = Global_Vars.MR
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    for i in range(Soln.shape[0]):
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_ATRUNet(CT, MR, Target, sol)
        Eval = evaluat_error(pred, Target)
        Fitn[i] = 1 / (Eval[0] + Eval[1]) + Eval[4]
    return Fitn
