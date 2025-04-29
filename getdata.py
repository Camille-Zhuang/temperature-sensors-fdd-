import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
# 1. 获取Token
def get_token():
    url = "http://119.167.138.16:6045/api/auth/login"
    headers = {"Content-Type": "application/json"}
    data = {
        "username": "qhsydyc@admin.com",
        "password": "sydyc123"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["token"]
    else:
        raise Exception(f"Failed to get token: {response.text}")
    
   # 2. 查询传感器数据
def query_sensor_data(token, start_ts, end_ts):
    url = "http://119.167.138.16:6045/api/plugins/telemetry/DEVICE/e7af93d0-6e4f-11ef-b65f-ab55797f86cb/values/timeseries"
    headers = {
        "X-Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    params = {
        "keys": "CH1_LDSW_T,CH1_LDRW_T,CH1_ZF_T,CH1_LN_T,CH1_LQSW_T,CH1_LQRW_T,CH1_S",
        "startTs": start_ts,
        "endTs": end_ts,
        "limit": 10000,
        "agg": "NONE",
        "interval": 600000
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to query data: {response.text}")

   # 3. 数据预处理
def preprocess_data(data):
    # 提取每个传感器的最新值
    processed_data = []
    for key in [ "CH1_LQRW_T","CH1_LQSW_T","CH1_LN_T", "CH1_S"]:
        if key in data:
            # 获取最新时间戳对应的数据
            latest_value = data[key][-1]["value"]
            processed_data.append(float(latest_value))
        else:
            # 如果某个传感器数据缺失，可以填充默认值或标记为缺失
            processed_data.append(0.0)
    #print(processed_data)
    # 将数据转换为DataFrame格式，行索引为["CH1_LQRW_T","CH1_LQSW_T","CH1_LN_T", "CH1_S"]
    df = pd.DataFrame([processed_data], columns=["CH1_LQRW_T","CH1_LQSW_T","CH1_LN_T", "CH1_S"])
    return df

  # 4. getdata主函数
def main_getdata():
    try:
        # 获取Token
        token = get_token()
        print("Token acquired successfully.")

        # 设置查询时间范围（例如：最近一天的数据）
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = end_ts - 24 * 60 * 60 * 1000  # 24小时前的时间戳

        # 查询传感器数据
        sensor_data = query_sensor_data(token, start_ts, end_ts)
        print("Sensor data queried successfully.")
        #打印出查询到的数据
        #print(json.dumps(sensor_data, indent=4, ensure_ascii=False))
        # 数据预处理
        input_data = preprocess_data(sensor_data)
        print("Data preprocessed successfully.")
        
        return input_data
        
        # 可以将结果保存到文件或数据库
        # np.savetxt("prediction_result.txt", prediction)

    except Exception as e:
        print(f"Error occurred: {e}")


#运行get_data主函数
if __name__ == "__main__":
   input_data =  main_getdata()