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
        self.model.build_model(self.config['Train']['batch_size'])
        self.model.summary()

        self.model.load_weights('./weights/Test/epoch_100_model.h5')

        # 2. Dataset
        self.dataset = DataGenerator(data_dir=self.config['Directory']['data_dir'],
                                        batch_size=1)
        
        self.test_dataset = self.dataset.get_testData(self.dataset.valid_data)
               
        # 3. Metric
        self.eval_mse = tf.keras.metrics.MeanSquaredError(name='eval_mse')
    
    
    @tf.function(jit_compile=True)
    def validation_step(self, x, y) -> tf.Tensor:
        pred = self.model(x, training=False)

        self.eval_mse.update_state(y, pred)
    
    def eval(self) -> None:
        timestamps = []
        input_list = []
        pred_list = []
        gt_list = []

        for i, (input_humidity, input_temp, gt_humidity, gt_temp, gt_time) in tqdm(enumerate(self.test_dataset),
                                                                          total=self.dataset.number_test):
            pred = self.model(input_temp)
            time = int(gt_time[0].numpy())
            time = datetime.fromtimestamp(time)

            timestamps.append(time)
            input_list.append(input_temp[0])
            pred_list.append(pred[0])
            gt_list.append(gt_temp[0])

            self.eval_mse.update_state(gt_temp, pred)

        print(len(pred_list))
        print(self.eval_mse.result())
        # 그래프 생성
        plt.figure(figsize=(10, 6))

        plt.plot(timestamps, input_list, color='g') # marker='o',
        plt.plot(timestamps, pred_list, color='r') # marker='o',
        plt.plot(timestamps, gt_list, color='b') # marker='o',


        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator())

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