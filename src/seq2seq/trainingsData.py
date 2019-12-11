import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from .alphabet import Alphabet

class TrainingsData:
    def __init__(self, input_tensor, label_tensor, alphabet,
                 batch_size = 64, validation_ratio = 0.1):
        self.alphabet = alphabet
        it, iv, lt, lv = train_test_split(input_tensor, label_tensor, test_size=validation_ratio)
        self.train_size = len(it)
        self.validation_size = len(iv)
        batch_size = min(len(it), batch_size)
        self.batch_size = batch_size
        
        train_data = tf.data.Dataset.from_tensor_slices((it, lt))
        train_data = train_data.shuffle(self.train_size)
        train_data = train_data.batch(batch_size, drop_remainder = True)
        self.training_data = train_data

        val_data = tf.data.Dataset.from_tensor_slices((iv, lv))
        val_data = val_data.shuffle(self.validation_size)
        val_data = val_data.repeat().batch(batch_size, drop_remainder = True)
        self.validation_data = val_data

        self.batches_per_epoch = max(1, len(it) // batch_size)

    @staticmethod
    def from_array(arr, alphabet = None):
        if alphabet is None:
            chars = { c for dataAndLabel in arr for dataOrLabel in dataAndLabel for c in dataOrLabel }
            alphabet = Alphabet.from_chars(chars)
        
        input, label = zip(*arr)

        def tensorfy(data):
            transformed = [alphabet.string_to_indices(str) for str in data]
            return pad_sequences(transformed, padding="post")
        
        return TrainingsData(tensorfy(input), tensorfy(label), alphabet)