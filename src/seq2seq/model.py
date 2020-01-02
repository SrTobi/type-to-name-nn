import tensorflow as tf
import time
import os

from .decoder import Decoder
from .encoder import Encoder
from .alphabet import Alphabet

class Model:
    def __init__(
            self, alphabet, units,
            checkpoint_dir = './training_checkpoints',
            checkpoint_prefix = 'ckpt'):
        self.alphabet = alphabet
        self.units = units
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.encoder = Encoder(alphabet.size, units)
        self.decoder = Decoder(alphabet.size, units)
        self.optimizer = tf.keras.optimizers.Adam()
        self.evaluatable = False
        self.checkpoint = tf.train.Checkpoint(optimizer = self.optimizer,
                                              encoder   = self.encoder,
                                              decoder   = self.decoder)

    def try_load(self, partial):
        latestchek = tf.train.latest_checkpoint(self.checkpoint_dir)
        if (latestchek is not None):
            result = self.checkpoint.restore(latestchek)
            if partial:
                result.expect_partial()
            self.evaluatable = True
        return latestchek
    
    def save(self):
        prefix = os.path.join(self.checkpoint_dir, self.checkpoint_prefix)
        self.checkpoint.save(file_prefix = prefix)

    
    def train(self, dataset, epochs = 1000, save_every_x_epoch = 10, do_validation = True):

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits = True,
            reduction   = 'none'
        )

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)


        start_idx = self.alphabet.char_to_ix(Alphabet.START)
        batch_size = dataset.batch_size
        batches_per_epoch = dataset.batches_per_epoch
        def initialize_hidden_state():
            return tf.zeros((batch_size, self.units))

        def train_step(inp, targ, do_validation):
            loss = 0
            enc_hidden = initialize_hidden_state()

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = self.encoder(inp, enc_hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([start_idx] * batch_size, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            if not do_validation:
                self.optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        for epoch in range(1, epochs + 1):
            start = time.time()

            total_loss = 0
            print(f"=== Epoch {epoch} ===")

            for (batch, (inp, targ)) in enumerate(dataset.training_data.take(batches_per_epoch)):
                batch_loss = train_step(inp, targ, False)
                total_loss += batch_loss

                if batch % 10 == 0:
                    print('Batch {} loss {:.5f}'.format(batch, batch_loss.numpy()))

            if do_validation:
                (val_input, val_target) = next(iter(dataset.validation_data))
                val_loss = train_step(val_input, val_target, True)

                print('Validation loss: {:.5f}'.format(val_loss))

            #if epoch > 100:
            #    translate(random.choice(raw_input)[1:-1])

            print('Epoch Loss {:.4f}'.format(total_loss / batches_per_epoch))
            print('Time taken for epoch: {:.2f} sec'.format(time.time() - start))


            # saving (checkpoint) the model every 3 epochs
            if epoch % save_every_x_epoch == 0:
                self.save()
                print("Model saved!")
            
            self.evaluatable = True
            
            print("\n")

            #result, input, _ = evaluate('3	Int	i')
            #print('%s -> %s' % (input, result))
            #result, input, _ = evaluate('true	Boolean	b')
            #print('%s -> %s' % (input, result))
            #result, input, _ = evaluate('"test"	String	str')
            #print('%s -> %s' % (input, result))

    def evaluate(self, input, max_output = 100):
        assert self.evaluatable
        attention_plot = []
        char_to_ix = self.alphabet.char_to_ix
        ix_to_char = self.alphabet.ix_to_char

        inputs = [char_to_ix(c) for c in input]
        #inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
        #                                                    maxlen=max_output,
        #                                                    padding='post')
        inputs = tf.convert_to_tensor([inputs])

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([char_to_ix(Alphabet.START)], 0)

        for _ in range(0, max_output):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot.append(attention_weights.numpy())

            predicted_idx = tf.argmax(predictions[0]).numpy()

            if predicted_idx == char_to_ix(Alphabet.END):
                break

            result += ix_to_char(predicted_idx)


            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_idx], 0)

        return result, attention_plot