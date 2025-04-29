import pandas as pd
import numpy as np
#判断input_data数据是否存在Complete Failure
def check_complete_failure(input_data):
    #初始化一个sensors_Status的DataFrame，并设置初始值都为0，大小为1行和input_data的列数-1，列名为input_data的列名，不要最后一列
    sensors_Status = pd.DataFrame(np.zeros((1, len(input_data.columns)-1)), columns=input_data.columns[:-1])
    #sensors_Status = sensors_Status.fillna(0)
    
    #除了最后一列CH1_S以外，input_data其余列均需要判断是否存在完全失效故障，完全故障定义为>0或大于45，若发生故障，则对应的sensors_Status列值为1，最后返回sensors_Status
    for i in range(len(input_data.columns)-1):
        if input_data.iloc[:, i].max() < 0 or input_data.iloc[:, i].max() > 45:
            sensors_Status.iloc[0, i] = 1
        else:
            sensors_Status.iloc[0, i] = 0
    return sensors_Status