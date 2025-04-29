from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

#导入本目录下的getdata.py文件
from getdata import main_getdata
from check_complete_failure import check_complete_failure

##定义PINNmodel

#4层
#===定义模型
class ANNModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,ouput_dim,num_layers):
        super(ANNModel,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU() #nn.Linear为线性关系，加上激活函数转为非线性
        self.fc2 = nn.Linear(hidden_dim, hidden_dim )
        #self.dropout = nn.Dropout(p=0.1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim )
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, ouput_dim )
        

    def forward(self,x):
        # 定义设备
        device = torch.device('cuda:0')
        out = self.fc1(x).to(device)
        #out = self.dropout(out).to(device)
        out = self.relu1(out).to(device)
        out = self.fc2(out).to(device)
        #out = self.dropout(out).to(device)
        out = self.relu2(out).to(device)
        out = self.fc3(out).to(device)
        out = self.relu3(out).to(device)
        out = self.fc4(out).to(device)
        #out = self.relu4(out).to(device)
        #out = self.fc5(out).to(device)
        return out



def check_slowdrift(input_data,Sensors_Status,n):
    # 定义设备
    device = torch.device('cuda:0')
    # 读取最值数据
    max_df = (pd.read_csv('max.csv')).T
    min_df = (pd.read_csv('min.csv')).T
    #区分X，Y的最值
    #max_value_x,max_value_y,min_value_x,min_value_y
    max_value_x = max_df.iloc[0, :n].values
    max_value_y = max_df.iloc[0, n:].values
    min_value_x = min_df.iloc[0, :n].values
    min_value_y = min_df.iloc[0, n:].values
    
    # 读取最优模型
    model = torch.load('model_best_case1_base.pth', map_location=device)
    #去掉input_data中最后一列
    input_data = input_data.iloc[:, :-1]
    ##归一化
    X_trans = np.array((input_data - min_value_x)/(max_value_x - min_value_x))
    #转化为tensor
    x_test = torch.from_numpy(X_trans).type(torch.Tensor).to(device)
    #验证模型
    pred = (model(x_test))
    pred_cpu = pred.cpu()
    pred_detached = pred_cpu.detach()
    df = pd.DataFrame(pred_detached.numpy())
    #反归一化
    pred_ori = df * (max_value_y - min_value_y) + min_value_y
    #输出结果，当pred_ori的绝对值大于2时，Sensors_Status为2，否则不变
    for i in range(len(pred_ori)):
        for j in range(len(pred_ori.columns)):
            if abs(pred_ori.iloc[i,j]) > 2:
                Sensors_Status.iloc[i,j] = 2
    return Sensors_Status

def main_slowdrift(input_data, Sensors_Status):
    # 确定回顾周期
    num_layers = 2
    lookback = 1
    hidden_dim = 30
    dropout = 0.5
    input_dim, output_dim = 3 , 3
    # 定义设备
    device = torch.device('cuda:0')
    model = ANNModel(input_dim=input_dim, hidden_dim=hidden_dim, ouput_dim=output_dim, num_layers=num_layers).to(device)
    # 检查慢漂移
    n = input_data.shape[1] - 1  # 假设最后一列是 CH1_S，不需要检查
    Sensors_Status = check_slowdrift(input_data, Sensors_Status, n)
    return Sensors_Status

if __name__ == "__main__":
    # 直接运行此脚本时会执行 main_slowdrift 函数
    Sensors_Status = main_slowdrift()
    #print(Sensors_Status)