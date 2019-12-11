import io

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
        
        self.trainings_data = TrainingsData.from_array(input_target)
        self.log(f"Training set has {self.trainings_data.train_size} entries")
        self.log(f"Validation set has {self.trainings_data.validation_size} entries")
        self.log(f"Batch size is {self.trainings_data.batch_size}")

        self.alphabet = self.trainings_data.alphabet
        self.log(f"Alphabet has {self.alphabet.size} chars")

        self.model = Model(self.alphabet, 500)
    
    def log(self, msg):
        if self.do_logging:
            print(msg)

    def train(self,
              load = True,
              save_every_x_epoch = 10):
        self.model.train(self.trainings_data, save_every_x_epoch = save_every_x_epoch)
        
    def evaluate(sample):
        return