import tensorflow as tf
import matplotlib.pyplot as plt
from utils.data_loader import DataGenerator
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

tf.executing_eagerly()
tf.config.run_functions_eagerly(True)
tf.config.optimizer.set_jit(False)

dataset = DataGenerator(data_dir='./data', batch_size=1)

train_data = dataset.get_trainData(dataset.train_data)

timestamp = []
x_temp_list = []
y_temp_list = []

for sample_data in train_data.take(dataset.number_train):
    x_hum, x_temp, y_hum, y_temp, time, _ = sample_data
    
    time = time[0].numpy().decode('utf-8')
    
    
    timestamp.append(time)
    x_temp_list.append(x_temp)
    y_temp_list.append(y_temp)
    print(sample_data)


# times = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in timestamp]
times = [datetime.fromisoformat(t.replace('Z', '+00:00')) + timedelta(hours=9) for t in timestamp]
print(times[0].timestamp())


# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(times, x_temp_list, color='r') # marker='o',
plt.plot(times, y_temp_list, color='b') # marker='o',


plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator())

# x축 라벨을 기울여서 표시
plt.gcf().autofmt_xdate()

plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature over Time')
plt.show()
