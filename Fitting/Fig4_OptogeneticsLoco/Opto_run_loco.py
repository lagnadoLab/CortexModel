import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import odeint, solve_ivp
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from lmfit import minimize, Parameter, Parameters, report_fit
from IPython import display  # optional unless you're using it
import os
import glob

max_nfev = 500

param_val = int(os.environ.get("PARAM_VAL", 1)) - 1 # Row id for getting parameters from dataframe (df_existing). 1 - default value to set if PARAM_VAL is not set.
print("Test param_val", param_val)

good_fits = pd.read_csv("good_fits_avg.csv")
iter_param = int(good_fits.iloc[param_val].values[0])# TODO: Here will be search by id in the array of good fits IDs
print("Test iter_param", iter_param)

matplotlib.use('Agg')

if not os.path.exists("reports"):
    os.makedirs("reports")

pdf_path = f"reports/report_{iter_param}.pdf"
ID_to_check = iter_param # This is parameter for ID of solution to check

pdf_pages = PdfPages(pdf_path)

def log_text(text):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 page
    ax.axis('off')
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=18, va='top')
    pdf_pages.savefig(fig)
    plt.close(fig)

def log_plot(fig):
    pdf_pages.savefig(fig)
    plt.close(fig)


def Step(t, stim = 5, inter = 10, amp = 1.0, base = 0, k = 0, delay = 0, decay_f = 0):
    """
    Feedback (FB) input:
    ------------------------------------
    Represented as a step function. Returns value of the input function in a time point t.
    ------------------------------------
    Parameters:
    
    t (running)             - time in seconds;
    stim (fixed) = 5 s      - time of the stimulus start;
    inter (fixed) = 10 s    - duration of the stimulus;
    amp (fixed) = 1.0 Hz    - amplitude of response;
    base (fixed) = 0 Hz     - baseline activity;
    k (fixed) = 0           - slope of the slow component. Used to be varied while testing linear depression or sensitization component in the FB input;
    delay (variable)        - delay of the FB input to the cell relative to the stimulus start;

    """
    
    if (t < stim + delay):
        h = base
    elif (t > stim + inter):
        h = amp * np.exp(-(t - stim - inter)*decay_f)
    else:
        h = amp*(t - stim - delay)*k*0.164745 + amp
    
    return h


def Sigm(t, stim = 5, inter = 10, ampl = 1.0, base = 0, rate = 1, delay = 0, decay_s = 1, dt = 0):
    """
    Slow modulation (SM) input:
    ------------------------------------
    Represented as a sigmoid function. Returns value of the input function in a time point t.
    ------------------------------------
    Parameters:
    
    t (running)               - time in seconds;
    stim (fixed) = 5 s        - time of the stimulus start;
    inter (fixed) = 10 s      - duration of the stimulus;
    ampl (fixed) = 1.0 Hz     - amplitude of response;
    base (fixed) = 0 Hz       - baseline activity;
    rate (variable)           - time-constant of the SM input;
    delay (variable)          - shift of the sigmoid center relative to stimulus start;
    decay_s (varaible)        - time-constant of the SM input exponential decay after the end of stimulation;

    """
    
    if (t < stim):
        h = base
    elif (t > stim + inter):
        h = (base + (ampl/(1 + np.exp((delay-inter - dt)/rate))))*np.exp(-(t - stim - inter)*decay_s) + base
    
    else:
        h = base + (ampl/(1 + np.exp((stim + delay - t)/rate))) #Actualy rate here is time constant, where 1/rate is actual rate
        
    
    return h


def expon(t, stim = 5, inter = 10, ampl = 1.5, base = 0, decay = 1, delay = 0, b = 0, decay_ff = 0, s_start = 0.1, k = 0.1):
    """
    Feedforward (FF) input:
    ------------------------------------
    Represented as a flat step function with fast exponential decay on the stimulus start and linear increase during 10 second period. 
    Returns value of the input function in a time point t.
    ------------------------------------
    Parameters:
    
    t (running)               - time in seconds;
    stim (fixed) = 5 s        - time of the stimulus start;
    inter (fixed) = 10 s      - duration of the stimulus;
    ampl (variable)           - amplitude of peak;
    base (fixed) = 1 Hz       - steady-state firing rate after fast exponential depression;
    decay (variable)          - time-constant of the fast exponential depression;
    delay (variable)          - delay of the FF input to the cell relative to the stimulus start;
    b (fixed) = 0 Hz          - baseline activity;
    decay_ff (varaible)       - time-constant of the FF input exponential decay after the end of stimulation;
    s_start (varaible)        - delay after stimulus when linear modulation starts;
    k (varaible)              - slope of the slow linear modulation;

    """
    
    if (t < stim + delay):
        h = b
    elif (t > stim + inter):
        h = (b + base + ampl*np.exp(-(inter-delay)*decay)+ (inter - delay - s_start)*k)*np.exp(-(t - stim - inter)*decay_ff)
    elif ((t >= stim + delay) and (t < stim + delay + s_start)):
        h = b + base + ampl*np.exp(-(t - stim - delay)*decay)
    else:
        h = b + base + ampl*np.exp(-(t - stim - delay)*decay)+ ((t - stim - delay - s_start)*k)
        
    return h


def model_step(t,
               y,
               w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18,
               tau_0, tau_1, tau_2, tau_3,
               threshold,
               power,
               q,
               i_0, i_1, i_2, i_3,
               r_1, decay, delay_1, delay_2, delay_3, ampl, base, decay_s, ampl_1, base_sigm, decay_f, decay_ff, s_start, k,
               opto_type, opto_param
               ):
    """
    Model basis:
    ------------------------------------
    Systems of first order differential equations that captures activity of populations without (df_xdt) and with (df_x_odt) optogenetic manipulations. 
    Returns an array of values of the derrivatives at a certain timepoint, calculated from values of functions in previous timepoint. 
    ------------------------------------
    Parameters:
    
    t (running)               - time in seconds;
    y (running)               - values of activities on the previous timepoint;
    w_x (variable)            - synaptic weights;
    tau_x (fixed)             - neurons time-constants;
    threshold (fixed) = 0 Hz  - minimum rectification value;
    power (fixed) = 2         - value of the power in the input-output function;
    q (fixed) = 1             - normalization coeficient in the input-output function;
    i_x (variable)            - baseline activity of neurons;
    ...
    
    """
    
            
    f_e, f_p, f_s, f_v, f_e_o, f_p_o, f_s_o, f_v_o = y
    
    ff_e = (min(max((i_0 + w_0 * f_e + w_1 * expon(t, ampl = ampl, base = base, decay = decay, delay = delay_1, decay_ff = decay_ff, s_start = s_start, k = k) + w_2 * Sigm(t, ampl = ampl_1, rate = r_1, delay = delay_2, decay_s = decay_s, base = base_sigm) + w_16 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) - w_3 * f_p - w_4 * f_s), threshold), 25))
    ff_p = (min(max((i_1 + w_17 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) + w_5 * f_e + w_6 * expon(t, ampl = ampl, base = base, decay = decay, delay = delay_1, decay_ff = decay_ff, s_start = s_start, k = k) + w_7 * Sigm(t, ampl = ampl_1, rate = r_1, delay = delay_2, decay_s = decay_s, base = base_sigm) - w_8 * f_p - w_9 * f_s), threshold), 25))
    ff_s = (min(max((i_2 + w_10 * f_e + w_11 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) - w_12 * f_v), threshold), 25))
    ff_v = (min(max((i_3 + w_18 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) + w_13 * f_e - w_14 * f_s +  w_15 * Sigm(t, ampl = ampl_1, rate = r_1, delay = delay_2, decay_s = decay_s, base = base_sigm)), threshold), 25))
        
    ff_e_o = (min(max((i_0 + w_0 * f_e_o + w_1 * expon(t, ampl = ampl, base = base, decay = decay, delay = delay_1, decay_ff = decay_ff, s_start = s_start, k = k) + w_2 * Sigm(t, ampl = ampl_1, rate = r_1, delay = delay_2, decay_s = decay_s, base = base_sigm) + w_16 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) - w_3 * f_p_o - w_4 * f_s_o), threshold), 25))
    ff_p_o = (min(max((i_1 + w_17 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) + w_5 * f_e_o + w_6 * expon(t, ampl = ampl, base = base, decay = decay, delay = delay_1, decay_ff = decay_ff, s_start = s_start, k = k) + w_7 * Sigm(t, ampl = ampl_1, rate = r_1, delay = delay_2, decay_s = decay_s, base = base_sigm) - w_8 * f_p_o - w_9 * f_s_o), threshold), 25)) #*invpow_pv_plast(t, stim = 5, inter = 10, base = 1, decay = 0.1, amp = 1, delta = 1) *Step_1(t, k=-0.2) Step_1(t, amp = 1.0, k=-0.13) exp_pv_plast(t, stim = 5, inter = 10, base = 1, decay = 0.01, amp = 1) *invpow_pv_plast(t, stim = 5, inter = 10, base = 1, decay = 0.1, amp = 1)
    ff_s_o = (min(max((i_2 + w_10 * f_e_o + w_11 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) - w_12 * f_v_o), threshold), 25))
    ff_v_o = (min(max((i_3 + w_18 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) + w_13 * f_e_o - w_14 * f_s_o +  w_15 * Sigm(t, ampl = ampl_1, rate = r_1, delay = delay_2, decay_s = decay_s, base = base_sigm)), threshold), 25))
    
        
    df_edt = ((q * ff_e ** power) - f_e) / tau_0
    df_pdt = ((q * ff_p ** power) - f_p) / tau_1
    df_sdt = ((q * ff_s ** power) - f_s) / tau_2
    df_vdt = ((q * ff_v ** power) - f_v) / tau_3

        
    if (t>=5): #and (t<=15):

        if opto_type == "PV_Arch":
            df_e_odt = ((q * (ff_e_o) ** power) - f_e_o) / tau_0
            df_p_odt = ((q * (ff_p_o / opto_param) ** power) - f_p_o) / tau_1  # /1.8 *1.7
            df_s_odt = ((q * (ff_s_o) ** power) - f_s_o) / tau_2  # /1.7 *1.4
            df_v_odt = ((q * (ff_v_o) ** power) - f_v_o) / tau_3

        elif opto_type == "PV_Chr":
            df_e_odt = ((q * (ff_e_o) ** power) - f_e_o) / tau_0
            df_p_odt = ((q * (ff_p_o * opto_param) ** power) - f_p_o) / tau_1  # /1.8 *1.7
            df_s_odt = ((q * (ff_s_o) ** power) - f_s_o) / tau_2  # /1.7 *1.4
            df_v_odt = ((q * (ff_v_o) ** power) - f_v_o) / tau_3

        elif opto_type == "SST_Arch":
            df_e_odt = ((q * (ff_e_o) ** power) - f_e_o) / tau_0
            df_p_odt = ((q * (ff_p_o) ** power) - f_p_o) / tau_1 #/1.8 *1.7
            df_s_odt = ((q * (ff_s_o/opto_param) ** power) - f_s_o) / tau_2 #/1.7 *1.4
            df_v_odt = ((q * (ff_v_o) ** power) - f_v_o) / tau_3

        elif opto_type == "SST_Chr":
            df_e_odt = ((q * (ff_e_o) ** power) - f_e_o) / tau_0
            df_p_odt = ((q * (ff_p_o) ** power) - f_p_o) / tau_1 #/1.8 *1.7
            df_s_odt = ((q * (ff_s_o*opto_param) ** power) - f_s_o) / tau_2 #/1.7 *1.4
            df_v_odt = ((q * (ff_v_o) ** power) - f_v_o) / tau_3
        
    else:
        
        df_e_odt = ((q * (ff_e_o) ** power) - f_e_o) / tau_0
        df_p_odt = ((q * (ff_p_o) ** power) - f_p_o) / tau_1
        df_s_odt = ((q * (ff_s_o) ** power) - f_s_o) / tau_2
        df_v_odt = ((q * (ff_v_o) ** power) - f_v_o) / tau_3
    

    dydt = [df_edt, df_pdt, df_sdt, df_vdt, df_e_odt, df_p_odt, df_s_odt, df_v_odt]
    
    return dydt


def exp_time(start, step, count, endpoint=False):
    """
    Experimental timepoints calculation:
    ------------------------------------
    Returns an array of values of the experimental timepoints. 
    ------------------------------------
    Parameters:
    
    start              - starting point;
    step               - value of time step of experimental recordings;
    count              - number of points;
    
    """
    stop = start+(step*count)
    return np.linspace(start, stop, count, endpoint=endpoint)


def odesol_step(tt, init, params, opto_type): 
    """
    Solves differential equation system defined in model_step() function.
    """
    y_init = init
    w_0 = params['w_0'].value
    w_1 = params['w_1'].value
    w_2 = params['w_2'].value
    w_3 = params['w_3'].value
    w_4 = params['w_4'].value
    w_5 = params['w_5'].value
    w_6 = params['w_6'].value
    w_7 = params['w_7'].value
    w_8 = params['w_8'].value
    w_9 = params['w_9'].value
    w_10 = params['w_10'].value
    w_11 = params['w_11'].value
    w_12 = params['w_12'].value
    w_13 = params['w_13'].value
    w_14 = params['w_14'].value
    w_15 = params['w_15'].value
    w_16 = params['w_16'].value
    w_17 = params['w_17'].value
    w_18 = params['w_18'].value
    
    tau_0 = params['tau_0'].value
    tau_1 = params['tau_1'].value
    tau_2 = params['tau_2'].value
    tau_3 = params['tau_3'].value
    threshold = params['threshold'].value
    power = params['power'].value
    q = params['q'].value
    i_0 = params['i_0'].value
    i_1 = params['i_1'].value
    i_2 = params['i_2'].value
    i_3 = params['i_3'].value
    ampl_1 = params['ampl_1'].value
    r_1 = params['r_1'].value
    delay_1 = params['delay_1'].value
    delay_2 = params['delay_2'].value
    delay_3 = params['delay_3'].value
    decay = params['decay'].value
    decay_s = params['decay_s'].value
    decay_f = params['decay_f'].value
    decay_ff = params['decay_ff'].value
    ampl = params['ampl'].value
    base = params['base'].value
    base_sigm = params['base_sigm'].value
    s_start = params['s_start'].value
    k = params['k'].value
    opto_param = params['opto_param'].value
   
    
    
    sol = solve_ivp(lambda t, y: model_step(t, y, 
                                            w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, 
                                            tau_0, tau_1, tau_2, tau_3,
                                            threshold,
                                            power, 
                                            q,
                                            i_0, i_1, i_2, i_3,
                                            r_1, decay, delay_1, delay_2, delay_3, ampl, base, decay_s, ampl_1, base_sigm, decay_f, decay_ff, s_start, k,
                                            opto_type, opto_param
                                           ), 
                    [tt[0], tt[-1]],
                    y_init,
                    method='RK45',
                    t_eval=tt,
                    #rtol = 1e-10, atol = 1e-12
                   )
    
    
    return sol


def simulate_step(tt, init, params, opto_type):
    """
    Model simulation:
    ------------------------------------
    Returns a Pandas DataFrame of timesteps and populational activity traces with and without optogenetic. 
    ------------------------------------
    Parameters:
    
    tt (running)        - current timepoint;
    init (fixed)        - intial conditions for differential equations;
    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    """
    

    
    sol = odesol_step(tt, init, params, opto_type)
    
    dd = np.vstack((sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], sol.y[5], sol.y[6], sol.y[7])).T
    sim = pd.DataFrame(dd, columns=['t', 'f_e', 'f_pv', 'f_sst', 'f_vip', 'f_e_o', 'f_pv_o', 'f_sst_o', 'f_vip_o'])
    
    return sim


# In[11]:


def residual_opto(params, tt, init, data_pc_opto, opto_type):
    """
    Residual function for fitting algorythms:
    ------------------------------------
    Returns flattened and concatenated array of residuals between model simulation and datapoints. 
    ------------------------------------
    Parameters:
    
    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    tt (running)        - current timepoint;
    init (fixed)        - intial conditions for differential equations;
    data_pc             - PC experimental data trace;
    data_pv             - PV experimental data trace;
    data_sst            - SST experimental data trace;
    data_vip            - VIP experimental data trace;
    
    """
    global count, max_nfev
    
    weights_pc_opto=np.float32(np.zeros_like(data_pc_opto))

    weights_pc_opto[25:93] = 1.
    
    model = simulate_step(tt, init, params, opto_type)
    pc_r_opto = (np.float32(np.array(model['f_e_o'].values - data_pc_opto))*weights_pc_opto).ravel() #/pc_all_err_new
    
    
    return pc_r_opto


def RMSE_full(params, init, data_pc, data_pv, data_sst, data_vip, opto_type):
    """
    Root:Mean Square Error (RMSE) calculation:
    ------------------------------------
    Returns RMSE value for full fit of 4 averages of populations: PC, PV, SST, VIP. 
    ------------------------------------
    Parameters:
    
    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    init (fixed)        - intial conditions for differential equations;
    data_pc             - PC experimental data trace;
    data_pv             - PV experimental data trace;
    data_sst            - SST experimental data trace;
    data_vip            - VIP experimental data trace;
    
    """
    model = simulate_step(t_exp, init, params, opto_type)
    sum = 0
    for i in range(len(data_pc)):
        sum += (model['f_e'].values[i] - data_pc[i])**2 + (model['f_pv'].values[i] - data_pv[i])**2 + (model['f_sst'].values[i] - data_sst[i])**2 + (model['f_vip'].values[i] - data_vip[i])**2

    sum_norm = np.sqrt((sum)/(len(data_pc)*4))

    return sum_norm

def RMSE_full_1(params, init, data_pc, data_pv, data_sst, data_vip, opto_type):
    """
    Root:Mean Square Error (RMSE) calculation:
    ------------------------------------
    Returns RMSE value for full fit of 4 averages of populations: PC, PV, SST, VIP. Calculated only for stimulus interval 
    ------------------------------------
    Parameters:
    
    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    init (fixed)        - intial conditions for differential equations;
    data_pc             - PC experimental data trace;
    data_pv             - PV experimental data trace;
    data_sst            - SST experimental data trace;
    data_vip            - VIP experimental data trace;
    
    """

    model = simulate_step(t_exp, init, params, opto_type)
    sum = 0
    for i in range(31,91):
        sum += (model['f_e'].values[i] - data_pc[i])**2 + (model['f_pv'].values[i] - data_pv[i])**2 + (model['f_sst'].values[i] - data_sst[i])**2 + (model['f_vip'].values[i] - data_vip[i])**2

    sum_norm = np.sqrt((sum)/(len(data_pc)*4))

    return sum_norm


def RMSE(params, init, data, type, opto_type):
    """
    Root Mean Square Error (RMSE) calculation:
    ------------------------------------
    Returns RMSE value for one defined fit. Used for optogenetic RMSE calculation
    ------------------------------------
    Parameters:
    
    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    init (fixed)        - intial conditions for differential equations;
    data                - Typically data for optogenetic effect on PCs;
    type                - Typically "f_e_o" - for optogenetic version of model.
    
    """

    model = simulate_step(t_exp, init, params, opto_type)
    sum = 0
    for i in range(31, 91):
        sum += (model[type].values[i] - data[i])**2

    sum = np.sqrt((sum)/len(data))

    return sum


def fit_and_save_opto(data_pc_opto, opto_type, rows):
    result = minimize(residual_opto, params, method='nelder', args=(t_exp, init, data_pc_opto, opto_type),
                      nan_policy='propagate', options={'adaptive': True}, max_nfev=max_nfev)

    RMSE_opto = RMSE(params, init, data_pc_opto, "f_e_o", opto_type)

    rows.append(
        {"ID": iter_param,
         "Parameter": "chisqr",
         "Value": f"{result.chisqr}",
         "Type": opto_type}
    )
    rows.append(
        {"ID": iter_param,
         "Parameter": "redchi",
         "Value": f"{result.redchi}",
         "Type": opto_type}
    )
    rows.append(
        {"ID": iter_param,
         "Parameter": "RMSE",
         "Value": f"{RMSE_opto}",
         "Type": opto_type}
    )
    rows.append(
        {"ID": iter_param,
         "Parameter": "opto_param",
         "Value": f"{result.params['opto_param'].value}",
         "Type": opto_type}
    )

    print(report_fit(result))

    model_opto = simulate_step(t_exp, init, result.params, opto_type)

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(t_exp, data_pc_opto, color='black', marker='o', label='f_pc(t)')
    ax[0].plot(t_exp, model_opto['f_e_o'].values, color='blue', label='fit', linewidth=3)
    ax[0].set_title(opto_type)
    ax[0].legend(loc='best')

    if "PV" in opto_type:
        ax[1].plot(t_exp, data_pv, color='black', marker='o', label='f_pv(t)')
        ax[1].plot(t_exp, model_opto['f_pv'].values, color='blue', label='fit', linewidth=3)
        ax[1].plot(t_exp, model_opto['f_pv_o'].values, color='red', label='fit_opto', linewidth=3)
        ax[1].set_title("PV")
        ax[1].legend(loc='best')
    else:
        ax[1].plot(t_exp, data_sst, color='black', marker='o', label='f_sst(t)')
        ax[1].plot(t_exp, model_opto['f_sst'].values, color='blue', label='fit', linewidth=3)
        ax[1].plot(t_exp, model_opto['f_sst_o'].values, color='red', label='fit_opto', linewidth=3)
        ax[1].set_title("SST")
        ax[1].legend(loc='best')

    ax[2].text(0.05, 0.95, f"opto param:\n{result.params['opto_param'].value}", transform=ax[2].transAxes,
               fontsize=22, va='top')
    log_plot(fig)

    return



#timepoint
dt = 0.164745

# PC_average (whole population)
pc_all = np.loadtxt(r'QA_Nor_PC_All_AvgSess1.txt')
# PC_average_standard_error
pc_all_err = np.loadtxt(r'QA_Nor_PC_All_SEMSess1.txt')

# PC during PV Arch opto
pc_opto_pv_arch = np.loadtxt(r'QA_Nor_AvgSess_PVArChT1_Opto.txt')
# PC during PV Chr opto
pc_opto_pv_chr = np.loadtxt(r'QA_Nor_AvgSess_PVChr1_Opto.txt')
# PC during SST Arch opto
pc_opto_sst_arch = np.loadtxt(r'QA_Nor_AvgSess_SSTArchT1_Opto.txt')
# PC during SST Chr opto
pc_opto_sst_chr = np.loadtxt(r'QA_Nor_AvgSess_SSTChr1_Opto.txt')

# Experimental time calculation
t_exp_1 = exp_time(start=0.0, step=dt, count=len(pc_all))

# PV_average (whole population)
pv = np.loadtxt(r'QA_Nor_PV_AvgSess.txt')
# PV_average_standard_error
pv_err = np.loadtxt(r'QA_Nor_PV_SEMSess.txt')

# Experimental time calculation
t_pv_exp_1 = exp_time(start=0.0, step=dt, count=len(pv))

# SST_average (whole population)
sst = np.loadtxt(r'QA_Nor_SST_AvgSess.txt')
# SST_average_standard_error
sst_err = np.loadtxt(r'QA_Nor_SST_SEMSess.txt')

# Experimental time calculation
t_sst_exp_1 = exp_time(start=0.0, step=dt, count=len(sst))

# VIP_average (whole population)
vip = np.loadtxt(r'QA_Nor_VIP_AvgSess.txt')
# VIP_average_standard_error
vip_err = np.loadtxt(r'QA_Nor_VIP_SEMSess.txt')

# Experimental time calculation
t_vip_exp_1 = exp_time(start=0.0, step=dt, count=len(vip))

# Inicialization of new arrays to cut traces for suitable range

t_exp_1_new = []
pv_new = []
sst_new = []
vip_new = []
pc_all_new = []
pc_all_err_new = []
pv_err_new = []
sst_err_new = []
vip_err_new = []
pc_opto_pv_arch_new = []
pc_opto_pv_chr_new = []
pc_opto_sst_chr_new = []
pc_opto_sst_arch_new = []

# Cutting the traces
for ind, time in np.ndenumerate(t_exp_1):
    if (time >= 5) and (time <= 25):
        t_exp_1_new.append(time - 5)
        pv_new.append(pv[ind])
        sst_new.append(sst[ind])
        vip_new.append(vip[ind])
        pc_all_new.append(pc_all[ind])
        pv_err_new.append(pv_err[ind])
        sst_err_new.append(sst_err[ind])
        vip_err_new.append(vip_err[ind])
        pc_opto_pv_arch_new.append(pc_opto_pv_arch[ind])
        pc_opto_pv_chr_new.append(pc_opto_pv_chr[ind])
        pc_opto_sst_chr_new.append(pc_opto_sst_chr[ind])
        pc_opto_sst_arch_new.append(pc_opto_sst_arch[ind])
        pc_all_err_new.append(pc_all_err[ind])
        

# Adjusting for a missing initial point after deconvolution
pv_new.insert(0, pv_new[0])
sst_new.insert(0, sst_new[0])
vip_new.insert(0, vip_new[0])
pc_all_new.insert(0, pc_all_new[0])
pv_err_new.insert(0, pv_err_new[0])
sst_err_new.insert(0, sst_err_new[0])
vip_err_new.insert(0, vip_err_new[0])
pc_opto_pv_arch_new.insert(0, pc_opto_pv_arch_new[0])
pc_opto_pv_chr_new.insert(0, pc_opto_pv_chr_new[0])
pc_opto_sst_chr_new.insert(0, pc_opto_sst_chr_new[0])
pc_opto_sst_arch_new.insert(0, pc_opto_sst_arch_new[0])
pc_all_err_new.insert(0, pc_all_err_new[0])

t_exp_1_new.append(t_exp_1_new[-1] + dt)

# Folder where your CSV files are
folder_path = os.path.join(".", "results_filtered")  # change this to your actual folder path
all_files = glob.glob(os.path.join(folder_path, "output_ID_*.csv"))

all_fits = pd.read_csv("combined_avg2.csv")

df_ID = all_fits[all_fits["ID"]==ID_to_check]


def value(name: str):
    return df_ID[name].values[0]

# This cell to set initial parameters and working parameters

t = np.float32(np.array(t_exp_1_new))
t_exp = t
threshold = np.float32(value("threshold")) 

power = np.float32(value("power"))
q = np.float32(value("q"))

ampl_1 = np.float32(value("ampl_1"))

r_1 = np.float32(value("r_1"))
delay_1 = np.float32(value("delay_1"))
delay_2 = np.float32(value("delay_2"))
delay_3 = np.float32(value("delay_3"))
decay = np.float32(value("decay"))
decay_s = np.float32(value("decay_s"))
decay_f = np.float32(value("decay_f"))
decay_ff = np.float32(value("decay_ff"))
ampl = np.float32(value("ampl"))
base = np.float32(value("base"))
base_sigm = np.float32(value("base_sigm"))
s_start = np.float32(value("s_start"))
k = np.float32(value("k"))

init = np.float32(np.array([0.055, 0.011, 0.017, 0.208, 0.055, 0.011, 0.017, 0.208]))
tau = np.float32(np.array([0.015, 0.0075, 0.019, 0.019]))
i = np.float32(np.array([np.float32(value("i_0")), 
                         np.float32(value("i_1")), 
                         np.float32(value("i_2")), 
                         np.float32(value("i_3"))]))
w = np.float32(np.array([
              np.float32(value("w_0")), # w_0  PC  -> PC        
              np.float32(value("w_1")), # w_1  FF  -> PC
              np.float32(value("w_2")), # w_2  SM   -> PC
              np.float32(value("w_3")), # w_3  PV -> PC
              np.float32(value("w_4")), # w_4  SST  -> PC
              np.float32(value("w_5")), # w_5  PC   -> PV
              np.float32(value("w_6")), # w_6  FF  -> PV
              np.float32(value("w_7")), # w_7  SM -> PV
              np.float32(value("w_8")), # w_8  PV   -> PV
              np.float32(value("w_9")), # w_9  SST  -> PV
              np.float32(value("w_10")), # w_10 PC -> SST
              np.float32(value("w_11")), # w_11 FB  -> SST
              np.float32(value("w_12")), # w_12 VIP  -> SST
              np.float32(value("w_13")), # w_13 PC  -> VIP
              np.float32(value("w_14")), # w_14 SST  -> VIP
              np.float32(value("w_15")), # w_15 SM  -> VIP
              np.float32(value("w_16")), # w_16 FB  -> PC
              np.float32(value("w_17")), # w_17 FB  -> PV
              np.float32(value("w_18"))]))# w_18 FB -> VIP



opto_type = "PV_Arch"  # "PV_Arch", "PV_Chr", "SST_Arch", "SST_Chr"
opto_param = np.float32(1.0)


data_pv = np.float32(np.array(pv_new)) 
data_sst = np.float32(np.array(sst_new)) 
data_vip = np.float32(np.array(vip_new)) 
data_pc_all = np.float32(np.array(pc_all_new))

data_pc_opto_pv_arch = np.float32(np.array(pc_opto_pv_arch_new))
data_pc_opto_pv_chr = np.float32(np.array(pc_opto_pv_chr_new))
data_pc_opto_sst_chr = np.float32(np.array(pc_opto_sst_chr_new))
data_pc_opto_sst_arch = np.float32(np.array(pc_opto_sst_arch_new))

pc_all_err_new = np.array(pc_all_err_new)
pv_err_new = np.array(pv_err_new)
sst_err_new = np.array(sst_err_new)
vip_err_new = np.array(vip_err_new)


count = 0


"""
Setting lmfit.Parameters() object with all parameters for the model,
their initial values, ranges and boolean varaible wether to vary them or not during fitting
"""

params = Parameters()
params.add('w_0', value = w[0] , vary = False, min = 0.05, max = 1.0)
params.add('w_1', value = w[1] , vary = False, min = 0.0, max = 4)
params.add('w_2', value = w[2] , vary = False, min = 0.0, max = 4)
params.add('w_3', value = w[3] , vary = False, min = 0.0, max = 4)
params.add('w_4', value = w[4] , vary = False, min = 0.0, max = 4)
params.add('w_5', value = w[5] , vary = False, min = 0.0, max = 4)
params.add('w_6', value = w[6] , vary = False, min = 0.0, max = 4)
params.add('w_7', value = w[7] , vary = False, min = 0.0, max = 4)
params.add('w_8', value = w[8] , vary = False, min = 0.0, max = 4)
params.add('w_9', value = w[9] , vary = False, min = 0.0, max = 4)
params.add('w_10', value = w[10] , vary = False, min = 0.0, max = 4)
params.add('w_11', value = w[11] , vary = False, min = 0.0, max = 4)
params.add('w_12', value = w[12] , vary = False, min = 0.0, max = 4)
params.add('w_13', value = w[13] , vary = False, min = 0.3, max = 4)
params.add('w_14', value = w[14] , vary = False, min = 0.0, max = 4)
params.add('w_15', value = w[15] , vary = False, min = 0.0, max = 4)
params.add('w_16', value = w[16] , vary = False, min = 0.0, max = 4)
params.add('w_17', value = w[17] , vary = False, min = 0.0, max = 4)
params.add('w_18', value = w[18] , vary = False, min = 0.0, max = 4)

params.add('tau_0', value = tau[0] , vary = False, min = 0.001, max = 0.03)
params.add('tau_1', value = tau[1] , vary = False, min = 0.001, max = 0.03)
params.add('tau_2', value = tau[2] , vary = False, min = 0.001, max = 0.03)
params.add('tau_3', value = tau[3] , vary = False, min = 0.001, max = 0.03)
params.add('threshold', value = threshold , vary = False, min = -np.inf, max = np.inf)
params.add('power', value = power , vary = False, min = 0.5, max = 2.5)
params.add('q', value = q, vary = False, min = 0.001, max = 2.5 )
params.add('i_0', value = i[0] , vary = False, min = 0.0, max = 0.7)
params.add('i_1', value = i[1] , vary = False, min = 0.0, max = 0.7)
params.add('i_2', value = i[2] , vary = False, min = 0.0, max = 0.7)
params.add('i_3', value = i[3] , vary = False, min = 0.0, max = 0.7)

params.add('ampl_1', value = ampl_1 , vary = False, min = 0.8, max = 1.5)
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
params.add('opto_param', value = opto_param, vary = True, min = 1.0, max = 15)
params


# Simulate a model with parameters defimed in params

model = simulate_step(t_exp, init, params, opto_type)


rmse_full = RMSE_full(params, init, data_pc_all, data_pv, data_sst, data_vip, opto_type)
rmse_full_txt = f"Whole trace RMSE: {rmse_full}"


rmse_full_stim = RMSE_full_1(params, init, data_pc_all, data_pv, data_sst, data_vip, opto_type)
rmse_full_stim_txt = f"Stimulus trace RMSE: {rmse_full_stim}"

log_text(f"Result for Parameter Set {ID_to_check}\n{rmse_full_txt}\n{rmse_full_stim_txt}")

fig, axes = plt.subplots(1, 4, figsize=(20, 6))  # 1 row, 4 columns

tick_size = 24
title_size = 24


axes[0].plot(t_exp, data_pc_all, color='black', marker='o', label='f_pc(t)')
axes[0].plot(t_exp, model['f_e'].values, color='blue', label='fit', linewidth=3)
axes[0].set_title("PC", size=title_size)
axes[0].tick_params(axis="both", labelsize=tick_size)

axes[1].plot(t_exp, data_pv, color='black', marker='o', label='f_pc(t)')
axes[1].plot(t_exp, model['f_pv'].values, color='blue', label='fit', linewidth=3)
axes[1].set_title("PV", size=title_size)
axes[1].tick_params(axis="both", labelsize=tick_size)

axes[2].plot(t_exp, data_sst, color='black', marker='o', label='f_pc(t)')
axes[2].plot(t_exp, model['f_sst'].values, color='blue', label='fit', linewidth=3)
axes[2].set_title("SST", size=title_size)
axes[2].tick_params(axis="both", labelsize=tick_size)

axes[3].plot(t_exp, data_vip, color='black', marker='o', label='f_pc(t)')
axes[3].plot(t_exp, model['f_vip'].values, color='blue', label='fit', linewidth=3)
axes[3].set_title("VIP", size=title_size)
axes[3].tick_params(axis="both", labelsize=tick_size)
log_plot(fig)


# Solution from paper

heat = pd.DataFrame()
heat[''] = ['PC', 'SST', 'PV', 'VIP_P', 'RMSE']

heat['PC'] = [0.082811034,
                  0.783707244,
                  0.20723919,
                  0.32398988,
                  None]

heat['SST'] = [0.631900957,
                  None,
                  0.250133287,
                  0.191125768,
                  None]

heat['PV'] = [1.955606639,
                   None,
                   1.510295591,
                   None,
                   None]

heat['VIP_P'] = [None,
                   0.14997032,
                   None,
                   None,
                   None]

heat['FF'] = [1.51969985,
                   None,
                   0.926368453,
                   None,
                   None]

heat['SM'] = [1.066189986,
                   None,
                   2.111694342,
                   0.737623002,
                   None]

heat['FB'] = [1.14600083,
                   0.594159328,
                   0.175807475,
                   0.158068172,
                   None]

heat['RMSE'] = [None,
                   None,
                   None,
                   None,
                   0.093223326]

heat.set_index('', inplace = True)

# Heatmap of resulting weights

heat_new_ = pd.DataFrame()
heat_new_[''] = ['PC', 'SST', 'PV', 'VIP_P', 'RMSE']

heat_new_['PC'] = [params["w_0"].value,
                  params["w_10"].value,
                  params["w_5"].value,
                  params["w_13"].value,
                  None]

heat_new_['SST'] = [params["w_4"].value,
                  None,
                  params["w_9"].value,
                  params["w_14"].value,
                  None]

heat_new_['PV'] = [params["w_3"].value,
                   None,
                   params["w_8"].value,
                   None,
                   None]

heat_new_['VIP_P'] = [None,
                   params["w_12"].value,
                   None,
                   None,
                   None]

heat_new_['FF'] = [params["w_1"].value,
                   None,
                   params["w_6"].value,
                   None,
                   None]

heat_new_['SM'] = [params["w_2"].value,
                   None,
                   params["w_7"].value,
                   params["w_15"].value,
                   None]

heat_new_['FB'] = [params["w_16"].value,
                   params["w_11"].value,
                   params["w_17"].value,
                   params["w_18"].value,
                   None]

heat_new_['RMSE'] = [None,
                   None,
                   None,
                   None,
                   rmse_full_stim]

heat_new_.set_index('', inplace = True)


heat_new_ = heat_new_.map(lambda x: np.nan if x is None else x)


fig, ax = plt.subplots(1, 2, figsize=(15, 4))
sns.heatmap(heat, ax=ax[0], annot = True, linewidth = 1, linecolor = 'black', fmt=".2f")
sns.heatmap(heat_new_, ax=ax[1], annot = True, linewidth = 1, linecolor = 'black', fmt=".2f")
ax[0].set_title("Old")
ax[0].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
ax[1].set_title("New")
ax[1].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
ax[1].set(xlabel="\nPresynaptic\n", ylabel="\nPostsynaptic\n")
ax[1].title.set_text("Fit Solution")
ax[1].xaxis.set_label_position("top")
log_plot(fig)

rows=[]

data_pc_opto = data_pc_opto_pv_arch
opto_type = "PV_Arch"

fit_and_save_opto(data_pc_opto, opto_type, rows)

data_pc_opto = data_pc_opto_pv_chr

opto_type = "PV_Chr"

fit_and_save_opto(data_pc_opto, opto_type, rows)

data_pc_opto = data_pc_opto_sst_arch 

opto_type = "SST_Arch"

fit_and_save_opto(data_pc_opto, opto_type, rows)

data_pc_opto = data_pc_opto_sst_chr 

opto_type = "SST_Chr"

fit_and_save_opto(data_pc_opto, opto_type, rows)

df_result = pd.DataFrame(rows)
df_result.to_csv(f"reports/opto_ID_{int(iter_param)}.csv", index=False)

pdf_pages.close()
