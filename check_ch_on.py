import pandas as pd
#判断冷机是否开启
def check_ch_on(input_data,sensors_Status):
    #chevk_ch_on函数，判断input_data数据是否存在CH1_S为1，即CH1_S为1，则返回True，否则返回False
    #input_data:输入数据，sensors_Status:传感器状态
    if input_data['CH1_S'].values[0] == 1:
        return True
    else:
        return False


