import numpy as np
from data import DataSets
from trim import target_trim,source_trim
from propensityscore import propensityscore
from calculate_overlapping_ import estimate_r
from calculate_weight import estimate_lamda
from composite_likelihood import etstimate_theta
data=DataSets()
target_ppscore,source_ppscore=propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_1_x[:,1:],source_y=data.training_historical_1_y)
target_data,split_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),target_ppscore)
source_data=source_trim(data.training_historical_1_x,data.training_historical_1_y.reshape(-1),source_ppscore,split_values)
est_r=np.empty(5)
est_lamda=np.empty(5)
N_1_k=np.empty(5)
est_theta=np.empty(5)
est_sigma=np.empty(5)
A=30
#r_k_jの推定
for i in range(0,5):
    target_condition=target_data['Group'] == i+1
    source_condition = source_data['Group'] == i+1
    target_group=target_data[target_condition]
    source_group=source_data[source_condition]
    est_r_k_j=estimate_r(target_group['Propensity_Score'].to_numpy(),source_group['Propensity_Score'].to_numpy())
    N_1_k[i]=len(source_group)
    est_r[i]=est_r_k_j[0]
sum_r=np.sum(est_r)    
#λ_kjの推定
for i in range(0,5):
    est_lamda[i]=estimate_lamda(sum_r=sum_r,r_k=est_r[i],A=A,N_k_j=N_1_k[i])

#θ_k_jの推定
for i in range(0,5):
    target_condition=target_data['Group'] == i+1
    source_condition = source_data['Group'] == i+1
    target_group=target_data[target_condition]
    source_group=source_data[source_condition]
    est_theta[i], est_sigma[i] = etstimate_theta(target_group['Outcomes'].to_numpy(), source_group['Outcomes'].to_numpy(), est_lamda[i])

print(est_r)
print(est_lamda)
print(np.sum(est_lamda))
print(est_theta)
print(est_sigma)
