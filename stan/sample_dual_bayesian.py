##################################################
### DRIFT DIFFUSION MODEL: BAYESIAN w/BIASED PRIOR & CHOICE PROCESS ###
##################################################


# Import necessary libraries
import pandas as pd
import numpy as np
import pystan
import pickle

# Import data
data = pd.read_pickle(r'../data/data-proc-pkl.pkl')

data2 = data[['participant','img_correct','response_correct','infer_resp_rt','feedback_correct','inf_bid_dv','block_loop_thisN']].copy()
data2 = data2[data2['block_loop_thisN'].notnull() & data2['img_correct'].notnull()]

model_data_rt = []
for p in data2['participant'].unique():
    pair_list_rt = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = list(data2[(data2['participant']==p) & (data2['img_correct']==i)].infer_resp_rt)
        pair_list_rt.append(x)
    model_data_rt.append(pair_list_rt)
missing = np.where(np.isnan(model_data_rt)==True)
for x in range(0,len(missing[0])):
    model_data_rt[missing[0][x]][missing[1][x]][missing[2][x]] = -1 # Replace missing response time data with -1, which we'll identify and exclude in the Stan model

# set invalid rt(<0.41) as 0.41 
for s in range(31):
    for p in range(20):
        for t in range(30):
            if model_data_rt[s][p][t] < 0.41 and model_data_rt[s][p][t] !=-1 :
                model_data_rt[s][p][t]=.41
                
model_data_correct = []
for p in data2['participant'].unique():
    pair_list_correct = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = map(int,list(data2[(data2['participant']==p) & (data2['img_correct']==i)].response_correct))
        pair_list_correct.append(list(x))    #list[list[i]][p]
    model_data_correct.append(pair_list_correct)

model_data_feedback = []
for p in data2['participant'].unique():
    pair_list_feedback = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = map(int,list(data2[(data2['participant']==p) & (data2['img_correct']==i)].feedback_correct))
        pair_list_feedback.append(list(x))
    model_data_feedback.append(pair_list_feedback)

model_data_bid_congruence = [] # Bid for correct item minus bid for incorrect item
for p in data2['participant'].unique():
    pair_list_bid_congruence = []
    for i in data2[data2['participant']==p]['img_correct'].unique():
        x = list(data2[(data2['participant']==p) & (data2['img_correct']==i)].inf_bid_dv)
        pair_list_bid_congruence.append(x)
    model_data_bid_congruence.append(pair_list_bid_congruence)

model_rt_mins = [] # Create a list of each subject's lowest response time
for p in data.participant.unique():
    model_rt_mins.append(data[data['participant']==p].infer_resp_rt.min())
# Replace subject E16's implausibly low minimum RT with the lowest RT from all other subjects' data
model_rt_mins[7] = 0.41 

model_data = {'NS': 31, 'NP': 20, 'NT': 30, 'correct': model_data_correct, 'feedback': model_data_feedback, 'bid_congruence': model_data_bid_congruence, 'rt': model_data_rt, 'rt_mins': model_rt_mins}

# set initial values for each parameter : indv? group?
# [0 for i in range(31)]
def initfunc():
    return dict(threshold_int_mean=1,threshold_int_sd=0.5, threshold_int=np.repeat(3,31),
                drift_rate_learning_mean=0.000001, drift_rate_learning_sd_unif=0.000001, drift_rate_learning_raw=np.repeat(0.000001, 31),
                cong_weight_prior_mean=0.000001, cong_weight_prior_sd_unif=0.000001, cong_weight_prior_raw=np.repeat(0.000001, 31),
                cong_weight_drift_bias_mean=0.000001, cong_weight_drift_bias_sd_unif=0.000001, cong_weight_drift_bias_raw=np.repeat(0.000001, 31), 
                ndt_1=0.000001, ndt_2=0.000001, ndt_3=0.000001, ndt_4=0.000001, ndt_5=0.000001, ndt_6=0.000001, ndt_7=0.000001, ndt_8=0.000001, ndt_9=0.000001, ndt_10=0.000001, 
                ndt_11=0.000001, ndt_12=0.000001, ndt_13=0.000001, ndt_14=0.000001, ndt_15=0.000001, ndt_16=0.000001, ndt_17=0.000001, ndt_18=0.000001, ndt_19=0.000001, ndt_20=0.000001,
                ndt_21=0.000001, ndt_22=0.000001, ndt_23=0.000001, ndt_24=0.000001, ndt_25=0.000001, ndt_26=0.000001, ndt_27=0.000001, ndt_28=0.000001, ndt_29=0.000001, ndt_30=0.0000001, 
                ndt_31=0.000001)

# Estimate models w/ pystan
# MODEL 1 : dual inference model
model_code_obj = pystan.StanModel(file='stan/model_dual_bayesian.stan.cpp', model_name='model_dual_bayesian') # Specific to model
fit = model_code_obj.sampling(data=model_data, iter=2000, chains=4, refresh=10, init=initfunc)
with open('pickles/model_dual_bayesian.pkl', 'wb') as f: # Specific to model
    pickle.dump(model_code_obj, f)

with open('pickles/fit_dual_bayesian.pkl', 'wb') as f: # Specific to model
    pickle.dump(fit, f)

print(fit)
