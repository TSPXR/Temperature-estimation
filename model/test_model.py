import tensorflow as tf

class TimeEncoder(tf.keras.Model):
    def __init__(self):
        super(TimeEncoder, self).__init__()
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Dense(32, use_bias=True, name='time_dense_1'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(64, use_bias=True, name='time_dense_2'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128, use_bias=True, name='time_dense_3'),
        ])

    def call(self, x, training=True):
        
        y = self.regressor(x, training=training)
        return y
    
class TempEncoder(tf.keras.Model):
    def __init__(self):
        super(TempEncoder, self).__init__()
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Dense(16, use_bias=True, name='temp_dense_1'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(32, use_bias=True, name='temp_dense_2'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(64, use_bias=True, name='temp_dense_3'),
        ])

    def call(self, x, training=True):
        y = self.regressor(x, training=training)
        return y

class DenseModule(tf.keras.Model):
    def __init__(self):
        super(DenseModule, self).__init__()
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, use_bias=True, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(2, use_bias=False, name='final_dense'),
        ])

    def call(self, x, training=True):
        y = self.dense(x, training=training)
        return y
    
class TSPSimulator(tf.keras.Model):
    def __init__(self, config):
        super(TSPSimulator, self).__init__()
        self.config = config
        self.batch_size = self.config['Train']['batch_size']

        self.time_encoder = TimeEncoder()       
        self.lstm_module = TempEncoder()
        self.dense = DenseModule()

    def call(self, time, x, training=True):
        # time = (batch, 10) x = (batch,)
        time_encode = self.time_encoder(time, training)
        temp_encode = self.lstm_module(x, training)
        concat_feature = tf.concat([time_encode, temp_encode], axis=-1)
        out = self.dense(concat_feature, training)
        return out

    def build_model(self, batch_size):
        humd_temp = tf.keras.layers.Input((2), batch_size=batch_size)
        time = tf.keras.layers.Input((10), batch_size=batch_size)
        outputs = self.call(time, humd_temp)
        self.inputs = [time, humd_temp]
        self.outputs = outputs

if __name__ == '__main__':
    config = {'Train':
              {'batch_size':8}
              }
    model = TSPSimulator(config=config)
    
    model.build_model(8)