import tensorflow as tf

activation = 'swish'

class TimeEncoder(tf.keras.Model):
    def __init__(self):
        super(TimeEncoder, self).__init__()
        self.kernel_size = 11
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, use_bias=False, kernel_size=self.kernel_size, padding='same', name='time_dense_1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv1D(64, use_bias=False, kernel_size=self.kernel_size, padding='same',name='time_dense_2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv1D(128, use_bias=True, kernel_size=self.kernel_size, padding='same',name='time_dense_3'),
        ])

    def call(self, x, training=True):
        
        y = self.regressor(x, training=training)
        return y
    
class TempEncoder(tf.keras.Model):
    def __init__(self):
        super(TempEncoder, self).__init__()
        self.kernel_size = 5
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Conv1D(16, use_bias=False, kernel_size=self.kernel_size, padding='same',name='temp_dense_1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv1D(32, use_bias=False, kernel_size=self.kernel_size, padding='same',name='temp_dense_2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv1D(64, use_bias=True, kernel_size=self.kernel_size, padding='same',name='temp_dense_3'),
        ])

    def call(self, x, training=True):
        y = self.regressor(x, training=training)
        return y
    

class HumidityEncoder(tf.keras.Model):
    def __init__(self):
        super(HumidityEncoder, self).__init__()
        self.kernel_size = 5

        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Conv1D(16, use_bias=False, kernel_size=self.kernel_size, padding='same',name='humidity_dense_1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv1D(32, use_bias=False, kernel_size=self.kernel_size, padding='same',name='humidity_dense_2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv1D(64, use_bias=True, kernel_size=self.kernel_size, padding='same',name='humidity_dense_3'),
        ])

    def call(self, x, training=True):
        y = self.regressor(x, training=training)
        return y
    

class LstmModule(tf.keras.Model):
    def __init__(self):
        super(LstmModule, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=256, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x, training=True):
        y = self.lstm(x, training=training)
        y = self.dropout(y)
        return y
      
class DenseModule(tf.keras.Model):
    def __init__(self):
        super(DenseModule, self).__init__()
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(256, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Dense(4, use_bias=True),
        ])

    def call(self, x, training=True):
        y = self.dense(x, training=training)
        return y
    
class TSPSimulator():
    def __init__(self, config):
        super(TSPSimulator, self).__init__()
        self.config = config
        self.batch_size = self.config['Train']['batch_size']

        self.time_encoder = TimeEncoder()       
        self.temp_encoder = TempEncoder()
        self.humd_encoder = HumidityEncoder()
        self.lstm = LstmModule()
        self.regressor = DenseModule()

    def call(self, inputs, training=True):
        """
            time : (batch, seq_len, 10)
            temp : (batch, seq_len, 1)
            hud : (batch, seq_len, 1)
        """
        time, temp, hud = inputs
        time_encode = self.time_encoder(time, training)
        temp_encode = self.temp_encoder(temp, training)
        hud_encode = self.humd_encoder(hud, training)

        concat_feature = tf.concat([time_encode, temp_encode, hud_encode], axis=-1)

        out = self.lstm(concat_feature, training)
        output = self.regressor(out, training)
        return output

    def build_model(self, batch_size):
        temp = tf.keras.layers.Input((10, 1), batch_size=batch_size)
        hud = tf.keras.layers.Input((10, 1), batch_size=batch_size)
        time = tf.keras.layers.Input((10, 10), batch_size=batch_size)
        inputs = [time, temp, hud]
        outputs = self.call(inputs)

        model = tf.keras.Model(inputs, outputs)
        return model
        
        

if __name__ == '__main__':
    config = {'Train':
              {'batch_size':8}
              }
    model = TSPSimulator(config=config)
    
    model.build_model(8)