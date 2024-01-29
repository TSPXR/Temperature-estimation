import tensorflow as tf
   
class LstmModule(tf.keras.Model):
    def __init__(self, rnn_hidden_size, rnn_dropout):
        super(LstmModule, self).__init__()
        # Define LSTM cells
        self.units = rnn_hidden_size
        self.rnn_dropout = rnn_dropout

        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Dense(32, use_bias=True, name='regression_dense_1'),
            tf.keras.layers.Dense(1, name='regression_final_dense')
        ])

    # Multi LSTM
    def call(self, x, training=True):
        x = tf.expand_dims(x, axis=-1)
        y = self.regressor(x, training=training)
        return y
    
class TSPSimulator(tf.keras.Model):
    def __init__(self, config):
        super(TSPSimulator, self).__init__()
        self.config = config
        self.batch_size = self.config['Train']['batch_size']
       
        self.lstm_module = LstmModule(self.config['Train']['rnn_hidden_size'],
                                   self.config['Train']['rnn_dropout'])

    def call(self, x, training=True):
        out = self.lstm_module(x, training)
        return out

    def build_model(self, batch_size):
        x = tf.keras.layers.Input((), batch_size=batch_size)
        outputs = self.call(x)
        self.inputs = [x]
        self.outputs = outputs