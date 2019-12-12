import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, enc_units):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = tf.one_hot(x, depth = self.vocab_size)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
