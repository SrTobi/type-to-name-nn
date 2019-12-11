class Alphabet:
    END = '¥'
    START='€'

    def __init__(self, char_to_ix, ix_to_char):
        self.size = len(char_to_ix)
        self.__char_to_ix = char_to_ix
        self.__ix_to_char = ix_to_char
    
    def char_to_ix(self, c):
        return self.__char_to_ix[c]
    
    def ix_to_char(self, ix):
        return self.__ix_to_char[ix]

    def string_to_indices(self, str):
        return [self.char_to_ix(c) for c in str]
    
    def indices_to_string(self, indices):
        return ''.join([self.ix_to_char(ix) for ix in indices])

    @staticmethod
    def from_chars(chars, additionals = [END, START]):
        chars = list(set(list(chars) + list(additionals)))
        chars.sort()
        char_to_ix = { ch:i for i,ch in enumerate(chars) }
        ix_to_char = { i:ch for ch,i in char_to_ix.items() }
        return Alphabet(char_to_ix, ix_to_char)

    @staticmethod
    def from_text(text, additionals = [END, START]):
        chars = set(text)
        return Alphabet.from_chars(chars, additionals)