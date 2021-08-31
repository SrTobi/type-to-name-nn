import io

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .seq2seq import *


class IdentifierPredictor:
    def __init__(self, file_path, log):
        self.do_logging = log
        self.log(f"Loading file: {file_path}")

        learn_file = io.open(file_path, encoding='UTF-8').read()
        lines = learn_file.strip().split('\n')

        self.log(f"Training file has {len(lines)} lines")

        input_target = []
        for (i, line) in enumerate(lines):
            #if i > 1000:
            #  break

            [expr, ty, id] = line.split('\t')
            if len(line) < 60:
                input_target.append([Alphabet.START + expr + '\t' + ty + Alphabet.END, id + Alphabet.END])
        
        print(input_target)

        self.trainings_data = TrainingsData.from_array(input_target)
        self.log(f"Training set has {self.trainings_data.train_size} entries")
        self.log(f"Validation set has {self.trainings_data.validation_size} entries")
        self.log(f"Batch size is {self.trainings_data.batch_size}")

        self.alphabet = self.trainings_data.alphabet
        self.log(f"Alphabet has {self.alphabet.size} chars")

        self.model = Model(self.alphabet, 500)

    def load(self, partial=False):

        res = self.model.try_load(partial)
        if res is None:
            self.log("Couldn't load checkpoint")
        else:
            self.log(f"Restored {res}")
    
    def log(self, msg):
        if self.do_logging:
            print(msg)

    def train(self,
              load = True,
              save_every_x_epoch = 10):
        self.model.train(self.trainings_data, save_every_x_epoch = save_every_x_epoch)
        
    def evaluate(self, input, draw_plot = False):
        input = Alphabet.START + input + Alphabet.END
        output, attention_mat = self.model.evaluate(input, max_output=15)
        print(f"Result: {output}")
        if draw_plot:
            self.plot_attention(attention_mat, input, output)

    def plot_attention(self, attention_mat, input, output):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention_mat, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + list(input), fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + list(output), fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()