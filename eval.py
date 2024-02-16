import tensorflow as tf
from tqdm import tqdm
import gc
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
import os
import matplotlib.dates as mdates
from utils.data_loader import DataGenerator
from model.test_model import TSPSimulator

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

class Evaluator(object):
    def __init__(self, config) -> None:
        self.config = config
        self._clear_session()
        self.configure_eval_ops()
        print('initialize')

    def _clear_session(self):
        """
            Tensorflow 계산 그래프의 이전 session 및 메모리를 초기화
        """
        tf.keras.backend.clear_session()
        _ = gc.collect()

    
    def configure_eval_ops(self) -> None:
        """
            학습 관련 설정
            1. Model
            2. Dataset
            3. Metric
        """
        # 1. Model
        self.model = TSPSimulator(config=self.config)
        self.model.built = True
        self.model.build_model(1)
        self.model.summary()

        self.model.load_weights('./weights/Test_2_WOBN_NEW/epoch_50_model.h5')

        # 2. Dataset
        self.dataset = DataGenerator(data_dir=self.config['Directory']['data_dir'],
                                        batch_size=1)
        
        self.test_dataset = self.dataset.get_testData(self.dataset.valid_data)
               
        # 3. Metric
        self.m2_abs_error = tf.keras.metrics.MeanAbsoluteError(name='M2_ABS_ERROR')
        self.m3_abs_error = tf.keras.metrics.MeanAbsoluteError(name='M3_ABS_ERROR')
    
    def eval(self) -> None:
        timestamps = []
        
        m1_temp_list = []
        m2_temp_list = []
        m3_temp_list = []
        pred_m2_list = []
        pred_m3_list = []

        for i, (m1, m2, m3) in tqdm(enumerate(self.test_dataset), total=self.dataset.number_test):
            m1_time, m1_humidity, m1_temp, m1_hit = m1
            _, m2_humidity, m2_temp, m2_hit = m2
            _, m3_humidity, m3_temp, m3_hit = m3

            x = tf.concat([m1_humidity, m1_temp], axis=-1)
            y_true = tf.concat([m2_temp, m3_temp], axis=-1)
            
            y_pred = self.model(m1_time, x, False)

            y_true = y_true[0]
            gt_m2 = tf.reduce_mean(y_true[:, 0]).numpy()
            gt_m3 = tf.reduce_mean(y_true[:, 1]).numpy()

            pred_m2 = tf.reduce_mean(y_pred[:, 0]).numpy()
            pred_m3 = tf.reduce_mean(y_pred[:, 1]).numpy()

            timestamps.append(str(i))

            pred_m2_list.append(pred_m2 * 100.)
            pred_m3_list.append(pred_m3 * 100.)
            m2_temp_list.append(gt_m2 * 100.)
            m3_temp_list.append(gt_m3 * 100.)

        self.m2_abs_error.update_state(m2_temp_list, pred_m2_list)
        self.m3_abs_error.update_state(m3_temp_list, pred_m3_list)

        print(len(pred_m2_list))
        print('M2 Sensor ABS ERROR :  ', float(self.m2_abs_error.result().numpy()) * 100, '°C')
        print('M3 Sensor ABS ERROR :  ', float(self.m3_abs_error.result().numpy()) * 100, '°C')
        # 그래프 생성
        plt.figure(figsize=(10, 6))

        interval = 1000
        sampled_timiestamps = timestamps[::interval]
        plt.plot(sampled_timiestamps, pred_m2_list[::interval], color='red', linestyle='dashed', label='M2 Prediction') # marker='o',
        plt.plot(sampled_timiestamps, pred_m3_list[::interval], color='blue', linestyle='dashed', label='M3 Prediction') # marker='o',
        plt.plot(sampled_timiestamps, m2_temp_list[::interval], color='darkred', label='M2 GT') # marker='o',
        plt.plot(sampled_timiestamps, m3_temp_list[::interval], color='darkblue', label='M3 GT') # marker='o',
        plt.legend(loc="upper right", prop={'size': 10})

        # x축 라벨을 기울여서 표시
        plt.gcf().autofmt_xdate()

        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature over Time')
        plt.show()


if __name__ == '__main__':
    # LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9" python trainer.py
    debug = False

    if debug:
        tf.executing_eagerly()
        tf.config.run_functions_eagerly(not debug)
        tf.config.optimizer.set_jit(False)
    else:
        tf.config.optimizer.set_jit(True)
        
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    with tf.device('/device:GPU:1'):
        # args = parser.parse_args()

        # Set random seed
        # SEED = 42
        # os.environ['PYTHONHASHSEED'] = str(SEED)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # tf.random.set_seed(SEED)
        # np.random.seed(SEED)

        trainer = Evaluator(config=config)
        trainer.eval()