#!python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import sys
import random


END = '¥'
START='€'
learn_file = io.open("all_vals", encoding='UTF-8').read()


def create_dataset():
    lines = learn_file.strip().split('\n')

    input_target = []
    for (i, line) in enumerate(lines):
      #if i > 1000:
      #  break

      [expr, ty, id] = line.split('\t')
      if len(line) < 60:
        input_target.append([START + expr + '\t' + ty + END, START + id + END])

    return zip(*input_target)


def max_length(tensor):
    return max(len(t) for t in tensor)

chars = list(set(learn_file)) + [END, START]
data_size, vocab_size = len(learn_file), len(chars) + 3
print("data has %d characters, %d unique." % (data_size, vocab_size))
char_to_ix = { ch:(i+1) for i,ch in enumerate(chars) }
ix_to_char = { (i+1):ch for i,ch in enumerate(chars) }
ix_to_char[0] = '#'

def tensorfy(list):
  return pad_sequences([[char_to_ix[c] for c in str] for str in list], padding="post")


raw_input, raw_target = create_dataset()
def load_dataset():
    # creating cleaned input, output pairs

    return tensorfy(raw_input), tensorfy(raw_target)



# Try experimenting with the size of that dataset
input_tensor, target_tensor = load_dataset()

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
print("Max input = %d" % max_length_inp)
print("Max target = %d" % max_length_targ)


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


def convert(tensor):
  for t in tensor:
    if t != 0:
      print ("%d ----> %s" % (t, ix_to_char[t]))


print ("Input; index to char mapping")
convert(input_tensor_train[0])
print ()
print ("Target Language; index to char mapping")
convert(target_tensor_train[0])



####################################################################################################üü
####################################################################################################üü
####################################################################################################üü



BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = max(1, len(input_tensor_train)//BATCH_SIZE)
units = 100

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(len(input_tensor_val))
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)



example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape



####################################################################################################üü



class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = tf.one_hot(x, depth=vocab_size)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


####################################################################################################üü


encoder = Encoder(vocab_size, units, BATCH_SIZE)

# sample input
#sample_hidden = encoder.initialize_hidden_state()
#sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
#print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
#print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))



####################################################################################################üü

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


####################################################################################################üü



attention_layer = BahdanauAttention(10)
#attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
#
#print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
#print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))




####################################################################################################üü


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = tf.one_hot(x, depth=vocab_size)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights





####################################################################################################üü

decoder = Decoder(vocab_size, units, BATCH_SIZE)

#sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1), dtype=tf.dtypes.int32, minval=1, maxval=vocab_size),
#                                      sample_hidden, sample_output)

#print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))




####################################################################################################üü


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

####################################################################################################üü

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


latestchek = tf.train.latest_checkpoint(checkpoint_dir)
if latestchek is not None:
  print("Load latest checkpoint (%s)" % latestchek)
  status = checkpoint.restore(latestchek)
  


####################################################################################################üü

@tf.function
def train_step(inp, targ, enc_hidden, validate):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([char_to_ix[START]] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  if not validate:
    optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss




####################################################################################################üü

def train():
  EPOCHS = 1000
  SAVE_EVERY_X_EPOCH = 10
  DO_VALIDATION = False

  for epoch in range(EPOCHS):
    start = time.time()

    total_loss = 0
    enc_hidden = encoder.initialize_hidden_state()

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
      batch_loss = train_step(inp, targ, enc_hidden, False)
      total_loss += batch_loss

      if batch % 10 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                      batch,
                                                      batch_loss.numpy()))
    # saving (checkpoint) the model every 3 epochs
    if (epoch + 1) % SAVE_EVERY_X_EPOCH == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      print("model saved!!!!!!!!!!!!!!!!!!!!!!!")

    if DO_VALIDATION:
      (val_input, val_target) = next(iter(val_dataset))
      val_loss = train_step(val_input, val_target, encoder.initialize_hidden_state(), True)

      print('Validation loss: {}'.format(val_loss))

    if epoch > 100:
      translate(random.choice(raw_input)[1:-1])

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    result, input, _ = evaluate('3	Int	i')
    print('%s -> %s' % (input, result))
    result, input, _ = evaluate('true	Boolean	b')
    print('%s -> %s' % (input, result))
    result, input, _ = evaluate('"test"	String	str')
    print('%s -> %s' % (input, result))





####################################################################################################üü



def evaluate(input):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    input = START + input + END
    inputs = [char_to_ix[i] for i in input]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([char_to_ix[START]], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += ix_to_char[predicted_id]

        if predicted_id == char_to_ix[END]:
            break

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, input, attention_plot


####################################################################################################


    # function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + list(sentence), fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + list(predicted_sentence), fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

####################################################################################################



def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result), :len(sentence)]
    plot_attention(attention_plot, sentence, result)

####################################################################################################


if len(sys.argv) >= 2:
  translate(sys.argv[1])
else:
  train()


