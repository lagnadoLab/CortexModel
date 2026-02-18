# Imports

import sys
import os
import pandas as pd
import numpy as np
import random
from scipy.integrate import odeint, solve_ivp
from scipy import stats
from lmfit import minimize, Parameter, Parameters, report_fit
from model_functions_loco_DepSen2 import *

########## DATA LOAD ##########################################################

# timepoint
dt = 0.164745
max_nfev = 1500
# PC_subpopulations (Dep, NA, Sen)
pc_sen = np.loadtxt(r'QA_Norx2_Dep_Avg1_Loco.txt')

# PC_subpopulations Standard Errors
pc_sen_err = np.loadtxt(r'QA_Norx2_Dep_SEM1_Loco.txt')
for i, v in np.ndenumerate(pc_sen_err):
    if v == 0:
        pc_sen_err[i] = stats.mode([i for i in pc_sen_err if i > 0]).mode

# PC_average (whole population)
pc_all = np.loadtxt(r'QA_Nor_PC_All_AvgSess1.txt')

# Experimental time calculation
t_exp_1 = exp_time(start=0.0, step=dt, count=len(pc_all))

# PV_average (whole population)
pv = np.loadtxt(r'QA_Nor_PV_AvgSess.txt')

# SST_average (whole population)
sst = np.loadtxt(r'QA_Nor_SST_AvgSess.txt')

# VIP_average (whole population)
vip = np.loadtxt(r'QA_Nor_VIP_AvgSess.txt')

input_file = "sobol_loco_filtered.csv"

if os.path.exists(input_file):
    df_existing = pd.read_csv(input_file)
    last_id = df_existing["ID"].max()
else:
    print("Sobol sample file didn't exist. Created a new one.")
    df_existing = pd.DataFrame()
    last_id = 0


###############################################################################

########## TRACE CUTTING ######################################################

# Inicialization of new arrays to cut traces for suitable range

t_exp_1_new = []
pc_sen_new = []
pv_new = []
sst_new = []
vip_new = []
pc_all_new = []
pc_sen_err_new = []

# Cutting the traces
for ind, time in np.ndenumerate(t_exp_1):
    if (time >= 5) and (time <= 25):
        t_exp_1_new.append(time - 5)
        pc_sen_new.append(pc_sen[ind])
        pv_new.append(pv[ind])
        sst_new.append(sst[ind])
        vip_new.append(vip[ind])
        pc_all_new.append(pc_all[ind])
        pc_sen_err_new.append(pc_sen_err[ind])

# Adjusting for a missing initial point after deconvolution
pc_sen_new.insert(0, pc_sen_new[0])
pv_new.insert(0, pv_new[0])
sst_new.insert(0, sst_new[0])
vip_new.insert(0, vip_new[0])
pc_all_new.insert(0, pc_all_new[0])
pc_sen_err_new.insert(0, pc_sen_err_new[0])

t_exp_1_new.append(t_exp_1_new[-1] + dt)
###############################################################################

########## VARIABLES AND CONSTANTS ############################################

scale = np.sqrt(np.float32(np.array(pc_sen_new))[30:90].mean()/np.float32(np.array(pc_all_new))[30:90].mean())
t = np.float32(np.array(t_exp_1_new))
t_exp = t

param_val = float(os.environ.get("PARAM_VAL", 1)) # Row id for getting parameters from dataframe (df_existing). 1 - default value to set if PARAM_VAL is not set.
print("Test param_val", param_val)
row = df_existing[df_existing['ID'] == param_val].iloc[0]
weights = row.drop("ID").values

threshold = np.float32(0)

power = np.float32(2.0)
q = np.float32(1.0)

ampl_1 = 1.0

r_1 = np.float32(1.713939)
delay_1 = np.float32(0.1650418)
delay_2 = np.float32(1.73305)
delay_3 = np.float32(0.3695274)
decay = np.float32(3.324089)
decay_s = np.float32(0.2685497)
decay_f = np.float32(0.8342502)
decay_ff = np.float32(2.152722)
ampl = np.float32(1.0)
base = np.float32(1.0)
base_sigm = np.float32(0)
s_start = np.float32(0.1835884)
k = np.float32(weights[19])

init = np.float32(np.array([0.055, 0.011, 0.017, 0.308, 0.0296]))
tau = np.float32(np.array([0.015, 0.0075, 0.019, 0.019]))
i = np.float32(np.array([0.26580502, 0.01447602, 0.11203215, 0.51341198]))

w = np.float32(np.array([
              weights[0],  # w_0  PC  -> PC
              weights[1],  # w_1  FF  -> PC
              weights[2],  # w_2  SM   -> PC
              weights[3],  # w_3  PV -> PC
              weights[4],  # w_4  SST  -> PC
              weights[5],  # w_5  PC   -> PV
              weights[6],  # w_6  FF  -> PV
              weights[7],  # w_7  SM -> PV
              weights[8],  # w_8  PV   -> PV
              weights[9],  # w_9  SST  -> PV
              weights[10],  # w_10 PC -> SST
              weights[11],  # w_11 FB  -> SST
              weights[12],  # w_12 VIP  -> SST
              weights[13],  # w_13 PC  -> VIP
              weights[14],  # w_14 SST  -> VIP
              weights[15],  # w_15 SM  -> VIP
              weights[16],  # w_16 FB  -> PC
              weights[17],  # w_17 FB  -> PV
              weights[18]]))  # w_18 FB -> VIP

i_d = np.array([i[0] * scale])

w_d = np.array([w[0] * scale, # w_d_0  PC  -> PC        
                w[1] * scale, # w_d_1  FF  -> PC
                w[2] * scale, # w_d_2  SM   -> PC
                w[16] * scale, # w_d_3  FB   -> PC
                w[3] * scale, # w_d_4  PV   -> PC
                w[4] * scale]) # w_d_5  SST -> PC


data_pc_sen = np.float32(np.array(pc_sen_new))
data_pv = np.float32(np.array(pv_new))
data_sst = np.float32(np.array(sst_new))
data_vip = np.float32(np.array(vip_new))
data_pc_all = np.float32(np.array(pc_all_new))

pc_sen_err_new = np.float32(np.array(pc_sen_err_new))
###############################################################################

########## SETTING PARAMETERS #################################################

"""
Setting lmfit.Parameters() object with all parameters for the model,
their initial values, ranges and boolean varaible wether to vary them or not during fitting
"""
params = Parameters()
params.add('w_0', value = w[0] , vary = False, min = 0.0, max = 5)
params.add('w_1', value = w[1] , vary = False, min = 0.0, max = 5)
params.add('w_2', value = w[2] , vary = False, min = 0.0, max = 5)
params.add('w_3', value = w[3] , vary = False, min = 0.0, max = 5)
params.add('w_4', value = w[4] , vary = False, min = 0.0, max = 5)
params.add('w_5', value = w[5] , vary = False, min = 0.0, max = 5)
params.add('w_6', value = w[6] , vary = False, min = 0.0, max = 5)
params.add('w_7', value = w[7] , vary = False, min = 0.0, max = 5)
params.add('w_8', value = w[8] , vary = False, min = 0.0, max = 5)
params.add('w_9', value = w[9] , vary = False, min = 0.0, max = 5)
params.add('w_10', value = w[10] , vary = False, min = 0.0, max = 5)
params.add('w_11', value = w[11] , vary = False, min = 0.0, max = 5)
params.add('w_12', value = w[12] , vary = False, min = 0.0, max = 5)
params.add('w_13', value = w[13] , vary = False, min = 0.0, max = 5)
params.add('w_14', value = w[14] , vary = False, min = 0.0, max = 5)
params.add('w_15', value = w[15] , vary = False, min = 0.0, max = 5)
params.add('w_16', value = w[16] , vary = False, min = 0.0, max = 5)
params.add('w_17', value = w[17] , vary = False, min = 0.0, max = 5)
params.add('w_18', value = w[18] , vary = False, min = 0.0, max = 5)

params.add('tau_0', value = tau[0] , vary = False, min = 0.001, max = 0.03)
params.add('tau_1', value = tau[1] , vary = False, min = 0.001, max = 0.03)
params.add('tau_2', value = tau[2] , vary = False, min = 0.001, max = 0.03)
params.add('tau_3', value = tau[3] , vary = False, min = 0.001, max = 0.03)
params.add('threshold', value = threshold , vary = False, min = -np.inf, max = np.inf)
params.add('power', value = power , vary = False, min = 0, max = 3.5)
params.add('q', value = q, vary = False, min = 0.001, max = 2.5 )
params.add('i_0', value = i[0] , vary = False, min = 0.0, max = 1.5)
params.add('i_1', value = i[1] , vary = False, min = 0.0, max = 1.5)
params.add('i_2', value = i[2] , vary = False, min = 0.0, max = 1.5)
params.add('i_3', value = i[3] , vary = False, min = 0.0, max = 1.5)
params.add('ampl_1', value = ampl_1 , vary = False, min = 0.0, max = 10)

params.add('i_d_0', value = i_d[0] , vary = False, min = 0, max = 5)
params.add('w_d_0', value = w_d[0] , vary = False, min = 0, max = 5)
params.add('w_d_1', value = w_d[1] , vary = True, min = 0.9 * w_d[1], max = 1.1 * w_d[1])
params.add('w_d_2', value = w_d[2] , vary = True, min = 0.9 * w_d[2], max = 1.1 * w_d[2])
params.add('w_d_3', value = w_d[3] , vary = True, min = 0.9 * w_d[3], max = 1.1 * w_d[3])
params.add('w_d_4', value = w_d[4] , vary = True, min = 0, max = 4)
params.add('w_d_5', value = w_d[5] , vary = True, min = 0, max = 4)

params.add('r_1', value = r_1, vary = False, min = 0, max = 5)
params.add('delay_1', value = delay_1, vary = False, min = 0, max = 1.5)
params.add('delay_2', value = delay_2, vary = False, min = 0, max = 25)
params.add('delay_3', value = delay_3, vary = False, min = 0, max = 3)
params.add('decay', value = decay, vary = False, min = 0, max = 10)
params.add('decay_s', value = decay_s, vary = False, min = 0, max = 10)
params.add('decay_f', value = decay_f, vary = False, min = 0, max = 10)
params.add('decay_ff', value = decay_ff, vary = False, min = 0, max = 10)
params.add('ampl', value = ampl, vary = False, min = 0, max = 5)
params.add('base', value = base, vary = False, min = 0, max = 5)
params.add('base_sigm', value = base_sigm, vary = False, min = 0, max = 2)
params.add('s_start', value = s_start, vary = False, min = 0, max = 4)
params.add('k', value = k, vary = False, min = 0.03, max = 2)
###############################################################################

########## FITTING Nelder #####################################################

result_nelder = minimize(residual_ad, params, method='nelder',
                         args=(t_exp, init, data_pc_sen, pc_sen_err_new),
                         nan_policy='propagate', options={'adaptive':True},
                         max_nfev=max_nfev)


RMSE_full_nelder = RMSE_full(result_nelder.params,
                             init, data_pc_all, data_pv, data_sst, data_vip, t_exp)

RMSE_full_nelder_stim = RMSE_full_stim(result_nelder.params,
                             init, data_pc_all, data_pv, data_sst, data_vip, t_exp)

# TODO: Write result in the output file.
rows = []

for name, param in params.items():
    rows.append(
        {"ID": param_val,
         "Parameter": name+"_init",
         "Value": f"{param.value:11.5f}",
         "Stderr": "N/A"}
    )

for name, param in result_nelder.params.items():
    stderr_val = f"{param.stderr:11.5f}" if param.stderr is not None else "N/A"
    rows.append(
        {"ID": param_val,
        "Parameter": name,
        "Value": f"{param.value:11.5f}",
        "Stderr": stderr_val}
    )

rows.append(
        {"ID": param_val,
        "Parameter": "chisqr",
        "Value": f"{result_nelder.chisqr}",
        "Stderr": "N/A"}
    )
rows.append(
        {"ID": param_val,
        "Parameter": "redchi",
        "Value": f"{result_nelder.redchi}",
        "Stderr": "N/A"}
    )

rows.append({
    "ID": param_val,
    "Parameter": "RMSE_full",
    "Value": f"{RMSE_full_nelder}",
    "Stderr": "N/A"
})
rows.append({
    "ID": param_val,
    "Parameter": "RMSE_full_stim",
    "Value": f"{RMSE_full_nelder_stim}",
    "Stderr": "N/A"
})


df_result = pd.DataFrame(rows)
#df_result = df_result.drop(columns=["Stderr"])
#df_wide = df_result.pivot(index="ID", columns="Parameter", values="Value").reset_index()

#df_wide.to_csv(f"results_rest/output_ID_{int(param_val)}.csv", index=False)

#if not os.path.exists(output_file):
#    df_wide.to_csv(output_file, index=False)
#    print(f"Created file with headers: {output_file}")
#else:
#    df_ = pd.read_csv(output_file)
#    df_final = pd.concat([df_, df_wide], ignore_index=True)
#    df_final.to_csv(output_file, index=False)

os.makedirs("results_rest", exist_ok=True)
df_result.to_csv(f"results_rest/output_ID_{int(param_val)}.csv", index=False)

print(report_fit(result_nelder))
print(RMSE_full_nelder)
print(RMSE_full_nelder_stim)
###############################################################################
