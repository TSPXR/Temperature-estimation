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
    print(sample_data)
    m1, m2, m3 = sample_data
    

    
    print('m1', m1)
    # print('m2', m2)
    # print('m3', m3)