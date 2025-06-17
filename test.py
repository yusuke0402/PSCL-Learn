import numpy as np
from data import DataSets
from trim import target_trim,source_trim
from propensityscore import propensityscore
data=DataSets()
target_ppscore,source_ppscore=propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_1_x[:,1:],source_y=data.training_historical_1_y)
target_data,split_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),target_ppscore)
source_data=source_trim(data.training_historical_1_x,data.training_historical_1_y.reshape(-1),source_ppscore,split_values)
print(source_data)
print(source_ppscore)
print(split_values)
