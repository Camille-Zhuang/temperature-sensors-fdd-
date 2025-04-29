from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

#导入本目录下的自定义py文件
from getdata import main_getdata
from check_complete_failure import check_complete_failure
from check_ch_on import check_ch_on
from check_slowdrift import main_slowdrift
from check_slowdrift import main_slowdrift, ANNModel


#温度传感器故障诊断流程函数
def main_temperature_failure_diagnosis():
    #1.获取数据
    input_data = main_getdata()
    #2.判断温度传感器是否存在完全失效故障
    Sensors_Status = check_complete_failure(input_data)
    #3.判断Sensors_Status中存在完全失效故障（即Sensors_Status存在value为1的值），则返回Sensors_Status
    if (Sensors_Status == 1).any(axis=1).any():
        return Sensors_Status
    else:
        #4.判断冷机开关机状态,当check_ch_on返回值为False时，则返回Sensors_Status
        if check_ch_on(input_data,Sensors_Status) == False:
            return Sensors_Status
        else:
            #5.判断温度传感器是否存在慢漂故障
            Sensors_Status = main_slowdrift(input_data, Sensors_Status)
        return Sensors_Status

if __name__ == "__main__":
    
    Sensors_Status = main_temperature_failure_diagnosis()
    print("Sensors Status Result:")
    print(Sensors_Status)