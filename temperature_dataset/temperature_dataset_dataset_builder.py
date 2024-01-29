"""temperature_dataset dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import os
from datetime import datetime, timedelta

def parse_sensor_data(filename):
    data = []
    with open(filename, 'r') as file:
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
          
            data.append({
                'time': time_delta,
                'humidity': float(humidity),
                'temperature': float(temperature),
                'heat_index': float(heat_index)
            })

    return data

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for temperature_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(temperature_dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'M1': tfds.features.FeaturesDict({
              'time': tfds.features.Tensor(shape=(), dtype=tf.int64),
              'humidity': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'temperature': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'heat_index': tfds.features.Tensor(shape=(), dtype=tf.float32)
            }),
            'M2': tfds.features.FeaturesDict({
              'time': tfds.features.Tensor(shape=(), dtype=tf.int64),
              'humidity': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'temperature': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'heat_index': tfds.features.Tensor(shape=(), dtype=tf.float32)
            }),
            'M3': tfds.features.FeaturesDict({
              'time': tfds.features.Tensor(shape=(), dtype=tf.int64),
              'humidity': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'temperature': tfds.features.Tensor(shape=(), dtype=tf.float32),
              'heat_index': tfds.features.Tensor(shape=(), dtype=tf.float32)
            }),
        }),
        disable_shuffling=True,
        supervised_keys=None,
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    archive_path = '../samples/sensor_data.zip'
    extracted_path = dl_manager.extract(archive_path)
    return {
        'train': self._generate_examples(root=extracted_path),
        'validation': self._generate_examples(root=extracted_path)
    }

  def _generate_examples(self, root):
    print()
    base = os.path.join(root)
    m1 = parse_sensor_data(base + '/m1.txt')
    m2 = parse_sensor_data(base + '/m2.txt')
    m3 = parse_sensor_data(base + '/m3.txt')

    sensor_data = {'M1': m1, 'M2': m2, 'M3': m3}

    for i, timestamp in enumerate(sensor_data['M1']):
        yield i, {
            'M1': sensor_data['M1'][i],
            'M2': sensor_data['M2'][i],
            'M3': sensor_data['M3'][i],
        }