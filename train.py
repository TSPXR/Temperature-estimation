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
        model_builder = TSPSimulator(config=self.config)
        
        self.model = model_builder.build_model(self.config['Train']['batch_size'])
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
                                                  weight_decay=self.config['Train']['weight_decay']
                                                  )# 

        # 4. Loss
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)
        
        # 5. Metric
        self.train_m2_rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='train_m2_rmse')
        self.train_m3_rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='train_m3_rmse')

        self.valid_m2_rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='valid_m2_rmse')
        self.valid_m3_rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='valid_m3_rmse')

        # 6. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = self.config['Directory']['log_dir'] + '/' + current_time + '_'
        self.train_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/valid')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        os.makedirs('{0}/{1}'.format(self.config['Directory']['weights'],
                                self.config['Directory']['exp_name']),
                                exist_ok=True)
    
    # @tf.function(jit_compile=True)
    def train_step(self, m1, m2, m3) -> tf.Tensor:
        with tf.GradientTape() as tape:
            # Forward pass
            m1_time, m1_humidity, m1_temp, _ = m1
            _, m2_humidity, m2_temp, _ = m2
            _, m3_humidity, m3_temp, _ = m3

            # m1_humidity = tf.expand_dims(m1_humidity, axis=-1)
            # m1_temp = tf.expand_dims(m1_temp, axis=-1)
            m2_gt = tf.concat([m2_temp, m2_humidity], axis=-1)
            m3_gt = tf.concat([m3_temp, m3_humidity], axis=-1)
            m2_pred, m3_pred = self.model([m1_time, m1_temp, m1_humidity], training=True)

            m2_loss = tf.keras.losses.mean_squared_error(m2_gt, m2_pred)
            m3_loss = tf.keras.losses.mean_squared_error(m3_gt, m3_pred)
            
            total_loss = m2_loss + m3_loss
            total_loss = tf.reduce_mean(total_loss)

            l2_losses = [0.000001 * tf.nn.l2_loss(v) for v in self.model.trainable_variables]
            l2_losses = tf.reduce_sum(l2_losses)

            total_loss += l2_losses
        
        # loss update
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metric
        self.train_m2_rmse_metric.update_state(m2_gt, m2_pred)
        self.train_m3_rmse_metric.update_state(m3_gt, m3_pred)

        return total_loss

    @tf.function(jit_compile=True)
    def validation_step(self, m1, m2, m3) -> tf.Tensor:
        m1_time, m1_humidity, m1_temp, _ = m1
        _, m2_humidity, m2_temp, _ = m2
        _, m3_humidity, m3_temp, _ = m3

        m2_gt = tf.concat([m2_temp, m2_humidity], axis=-1)
        m3_gt = tf.concat([m3_temp, m3_humidity], axis=-1)
        m2_pred, m3_pred = self.model([m1_time, m1_temp, m1_humidity], training=False)

        # Update metric
        self.valid_m2_rmse_metric.update_state(m2_gt, m2_pred)
        self.valid_m3_rmse_metric.update_state(m3_gt, m3_pred)
    
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
            for _, (m1, m2, m3) in enumerate(train_tqdm):
                epoch_loss = self.train_step(m1, m2, m3)

            with self.train_summary_writer.as_default():
                tf.summary.scalar(self.train_m2_rmse_metric.name, self.train_m2_rmse_metric.result(), step=epoch)
                tf.summary.scalar(self.train_m3_rmse_metric.name, self.train_m3_rmse_metric.result(), step=epoch)
                tf.summary.scalar('epoch_loss', tf.reduce_mean(epoch_loss).numpy(), step=epoch)
            
            # Validation
            valid_tqdm = tqdm(self.test_dataset, total=self.dataset.number_test)
            valid_tqdm.set_description('Validation || ')
            for _, (m1, m2, m3) in enumerate(valid_tqdm):
                self.validation_step(m1, m2, m3)

            with self.valid_summary_writer.as_default():
                tf.summary.scalar(self.valid_m2_rmse_metric.name, self.valid_m2_rmse_metric.result(), step=epoch)
                tf.summary.scalar(self.valid_m3_rmse_metric.name, self.valid_m3_rmse_metric.result(), step=epoch)

            if epoch % 5 == 0:
                # self.model.save_weights('./{0}/epoch_{1}_model.h5'.format(self.config['Directory']['weights'], epoch))

                self.model.save_weights('{0}/{1}/epoch_{2}_model.h5'.format(self.config['Directory']['weights'],
                                                                            self.config['Directory']['exp_name'],
                                                                            epoch))
            # Log epoch loss

            print(f'\n \
                    train_M2 : {self.train_m2_rmse_metric.result()}, train_M3 : {self.train_m3_rmse_metric.result()}, \n \
                    valid_M2 : {self.valid_m2_rmse_metric.result()}, valid_M3 : {self.valid_m3_rmse_metric.result()} \n')
            
            # clear_session()
            self.train_m2_rmse_metric.reset_states()
            self.train_m3_rmse_metric.reset_states()
            
            self.valid_m2_rmse_metric.reset_states()
            self.valid_m3_rmse_metric.reset_states()

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