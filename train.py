import tensorflow as tf
from tqdm import tqdm
import gc
from datetime import datetime
import yaml
import os
from utils.data_loader import DataGenerator
from model.test_model import TSPSimulator

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

class Trainer(object):
    def __init__(self, config) -> None:
        self.config = config
        self._clear_session()
        self.configure_train_ops()
        print('initialize')

    def _clear_session(self):
        """
            Tensorflow 계산 그래프의 이전 session 및 메모리를 초기화
        """
        tf.keras.backend.clear_session()
        _ = gc.collect()

    
    def configure_train_ops(self) -> None:
        """
            학습 관련 설정
            1. Model
            2. Dataset
            3. Optimizer
            4. Loss
            5. Metric
            6. Logger
        """
        # 1. Model
        self.model = TSPSimulator(config=self.config)
        self.model.built = True
        self.model.build_model(self.config['Train']['batch_size'])
        self.model.summary()
        # self.model.load_weights('./weights/epoch_100_model.h5')

        # 2. Dataset
        self.dataset = DataGenerator(data_dir=self.config['Directory']['data_dir'],
                                        batch_size=self.config['Train']['batch_size'])
        
        self.train_dataset = self.dataset.get_trainData(self.dataset.train_data)
        self.test_dataset = self.dataset.get_testData(self.dataset.valid_data)
        
        # 3. Optimizer
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                        self.config['Train']['epoch'],
                                                                         self.config['Train']['init_lr'] * 0.01,
                                                                         power=0.9)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['Train']['init_lr'],
                                                  )# weight_decay=self.config['Train']['weight_decay']

        # 4. Loss
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)
        
        # 5. Metric
        self.train_rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='train_rmse')
        self.valid_rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='valid_rmse')

        # 6. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = self.config['Directory']['log_dir'] + '/' + current_time + '_'
        self.train_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/valid')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        os.makedirs('{0}/{1}'.format(self.config['Directory']['weights'],
                                self.config['Directory']['exp_name']),
                                exist_ok=True)
    
    @tf.function(jit_compile=True)
    def train_step(self, x, y) -> tf.Tensor:
        with tf.GradientTape() as tape:
            # Forward pass
            pred = self.model(x, training=True)

            angle_loss = self.mse_loss(y, pred)
            
            total_loss = tf.reduce_mean(angle_loss)
        
        # loss update
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metric
        self.train_rmse_metric.update_state(y, pred)

        return total_loss

    @tf.function(jit_compile=True)
    def validation_step(self, x, y) -> tf.Tensor:
        pred = self.model(x, training=False)

        self.valid_rmse_metric.update_state(y, pred)
    
    def train(self) -> None:        
        for epoch in range(self.config['Train']['epoch']):
            # Get parameter status
            
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr
            
            train_tqdm = tqdm(self.train_dataset, total=self.dataset.number_train)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} || LR : {1} ||'.format(epoch, 
                                                                                         round(float(self.optimizer.learning_rate.numpy()), 8)))
            for _, (input_humidity, input_temp, gt_humidity, gt_temp) in enumerate(train_tqdm):
                epoch_loss = self.train_step(input_temp, gt_temp)

            with self.train_summary_writer.as_default():
                tf.summary.scalar(self.train_rmse_metric.name, self.train_rmse_metric.result(), step=epoch)
                tf.summary.scalar('epoch_loss', tf.reduce_mean(epoch_loss).numpy(), step=epoch)
            
            # Validation
            valid_tqdm = tqdm(self.test_dataset, total=self.dataset.number_test)
            valid_tqdm.set_description('Validation || ')
            for _, (input_humidity, input_temp, gt_humidity, gt_temp) in enumerate(valid_tqdm):
                self.validation_step(input_temp, gt_temp)

            with self.valid_summary_writer.as_default():
                tf.summary.scalar(self.valid_rmse_metric.name, self.valid_rmse_metric.result(), step=epoch)

            if epoch % 5 == 0:
                # self.model.save_weights('./{0}/epoch_{1}_model.h5'.format(self.config['Directory']['weights'], epoch))

                self.model.save_weights('{0}/{1}/epoch_{2}_model.h5'.format(self.config['Directory']['weights'],
                                                                            self.config['Directory']['exp_name'],
                                                                            epoch))
            # Log epoch loss
            print(f'\n \
                    train_RMSE : {self.train_rmse_metric.result()}, \n \
                    valid_RMSE : {self.valid_rmse_metric.result()}')

            # clear_session()
            self.train_rmse_metric.reset_states()
            self.valid_rmse_metric.reset_states()

            self._clear_session()

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

        trainer = Trainer(config=config)
        trainer.train()