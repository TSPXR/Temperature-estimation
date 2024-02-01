import tensorflow_datasets as tfds
import tensorflow as tf
import math
from typing import Union
from datetime import datetime, timedelta
import random

AUTO = tf.data.experimental.AUTOTUNE
# AUTO = 24

class TFDataLoadHandler(object):
    def __init__(self):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            dataset_name (str)   : Tensorflow dataset name (e.g: 'citiscapes')
        
        """
        self.dataset_name = 'temperature_dataset'
        self.data_dir = './data'
        self.train_data, self.valid_data = self.__load_custom_dataset()
        
    def __load_custom_dataset(self) -> Union[tf.data.Dataset, tf.data.Dataset]:
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train')
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='validation')
        return train_data, valid_data
    
class DataLoadHandler(object):
    def __init__(self) -> None:
        self.file_names = ['./samples/m1.txt', './samples/m2.txt', './samples/m3.txt']
        self.create_dataset(self.file_names)

    def create_dataset(self, file_names):
        # 각 파일의 데이터를 딕셔너리로 로드
        train_dataset = {}
        valid_dataset = {}

        for idx, file_name in enumerate(file_names):
            train_data, valid_data = self.parse_sensor_data(file_name, 0.2)
            train_dataset[f'M{idx+1}'] = tf.data.Dataset.from_generator(
                lambda data=train_data: (d for d in data),
                output_types={'month': tf.float32,
                              'day': tf.float32,
                              'hour': tf.float32,
                              'minute': tf.float32,
                              'second': tf.float32,
                              'humidity': tf.float32,
                              'temperature': tf.float32,
                              'heat_index': tf.float32}
            )

            valid_dataset[f'M{idx+1}'] = tf.data.Dataset.from_generator(
                lambda data=valid_data: (d for d in data),
             output_types={'month': tf.float32,
                              'day': tf.float32,
                              'hour': tf.float32,
                              'minute': tf.float32,
                              'second': tf.float32,
                              'humidity': tf.float32,
                              'temperature': tf.float32,
                              'heat_index': tf.float32}
            )

        # 모든 데이터셋을 하나로 병합
        self.train_data = tf.data.Dataset.zip(train_dataset)
        self.valid_data = tf.data.Dataset.zip(valid_dataset)

    def parse_sensor_data(self, file_name, split_ratio):
        data = []
        with open(file_name, 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                timestamp = parts[0]
                sensor_data = parts[1].split('::: ')[1]

                # 습도, 온도, 체감온도
                humidity = sensor_data.split('Humidity: ')[1].split('%')[0]
                temperature = sensor_data.split('Temperature: ')[1].split('°C')[0]
                heat_index = sensor_data.split('Heat index: ')[1].split('°C')[0]

                time_delta = datetime.fromisoformat(timestamp.replace('Z', '+00:00')) + timedelta(hours=9)
                time_delta = time_delta.timestamp()
                time_delta_datetime = datetime.fromtimestamp(time_delta)

                data.append({
                    'month': float(time_delta_datetime.month),
                    'day': float(time_delta_datetime.day),
                    'hour': float(time_delta_datetime.hour),
                    'minute': float(time_delta_datetime.minute),
                    'second': float(time_delta_datetime.second),
                    'humidity': float(humidity),
                    'temperature': float(temperature),
                    'heat_index': float(heat_index)
                })
        data_size = int(len(data) * split_ratio)
        train_data = data[data_size:]
        valid_data = data[:data_size]
        return train_data, valid_data

class DataGenerator(DataLoadHandler):
    def __init__(self, data_dir: str, batch_size: int):
        """
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            batch_size   (int)   : Batch size
        """
        # Configuration
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.number_train = self.train_data.reduce(0, lambda x, _: x + 1).numpy() // self.batch_size
        self.number_test = self.valid_data.reduce(0, lambda x, _: x + 1).numpy() // self.batch_size

    @tf.function(jit_compile=True)
    def cyclic_encoding(self, value, max_value):
        sin = tf.sin(2 * tf.experimental.numpy.pi * value / max_value)
        cos = tf.cos(2 * tf.experimental.numpy.pi * value / max_value)
        return [sin, cos]

    @tf.function(jit_compile=True)
    def normalize_data(self, sensor_data):
        month = sensor_data['month']
        day = sensor_data['day']
        hour = sensor_data['hour']
        minute = sensor_data['minute']
        second = sensor_data['second']
        humidity = sensor_data['humidity']
        temperature = sensor_data['temperature']
        heat_index = sensor_data['heat_index']

        month_cycle = self.cyclic_encoding(month, 12)
        day_cycle = self.cyclic_encoding(day, 31)
        hour_cycle = self.cyclic_encoding(hour, 24)
        minute_cycle = self.cyclic_encoding(minute, 60)
        second_cycle = self.cyclic_encoding(second, 60)
        time = tf.concat([month_cycle, day_cycle, hour_cycle, minute_cycle, second_cycle], axis=-1)
        humidity /= 100.
        humidity = tf.expand_dims(humidity, axis=-1)
        
        temperature /= 100.
        temperature = tf.expand_dims(temperature, axis=-1)

        heat_index /= 100.
        heat_index = tf.expand_dims(heat_index, axis=-1)
        return (time, humidity, temperature, heat_index)

    @tf.function(jit_compile=True)
    def parsing_sensor_data(self, sensor_data: dict):

        normalized_data = self.normalize_data(sensor_data)
        return normalized_data

    @tf.function(jit_compile=True)
    def prepare_data(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
         'M1': tfds.features.FeaturesDict({
              'time': tfds.features.Text(),
              'humidity': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'temperature': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'heat_index': tfds.features.Tensor(shape=(), dtype=tf.float32)
            }),
            'M2': tfds.features.FeaturesDict({
              'time': tfds.features.Text(),
              'humidity': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'temperature': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'heat_index': tfds.features.Tensor(shape=(), dtype=tf.float32)
            }),
            'M3': tfds.features.FeaturesDict({
              'time': tfds.features.Text(),
              'humidity': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'temperature': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'heat_index': tfds.features.Tensor(shape=(), dtype=tf.float32)
            }),

            Load RGB images and segmentation labels from the dataset.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().

            Returns:
                (img, labels) (dict) : Returns the image and label extracted from sample as a key value.
        """
        M1 = sample['M1']
        M2 = sample['M2']
        M3 = sample['M3']

        input_m1 = self.parsing_sensor_data(sensor_data=M1)
        gt_m1 = self.parsing_sensor_data(sensor_data=M2)
        gt_m3 = self.parsing_sensor_data(sensor_data=M3)

        return (input_m1, gt_m1, gt_m3)

    @tf.function(jit_compile=True)
    def preprocess(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Dataset mapping function to apply to the train dataset.
            Various methods can be applied here, such as image resizing, random cropping, etc.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().
            
            Returns:
                (img, labels) (dict) : tf.Tensor
        """
        m1, m2, m3 = self.prepare_data(sample)
        
        return (m1, m2, m3)
    
    def get_trainData(self, train_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                train_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                train_data    (tf.data.Dataset)  : Apply data augmentation, batch, and shuffling
        """    
        train_data = train_data.shuffle(1024, reshuffle_each_iteration=True)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        train_data = train_data.prefetch(AUTO)
        
        # train_data = train_data.repeat()
        return train_data

    def get_testData(self, valid_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                valid_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                valid_data    (tf.data.Dataset)  : Apply data resize, batch, and shuffling
        """    
        valid_data = valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        valid_data = valid_data.batch(self.batch_size, drop_remainder=True)
        valid_data = valid_data.prefetch(AUTO)
        return valid_data
    
if __name__ == '__main__':
    print(AUTO)