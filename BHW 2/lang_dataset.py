import numpy as np
from torch.utils.data import Dataset


class LangDataset(Dataset):
    def __init__(self, source_file_de, vocab_de, max_len, source_file_en=None,
                 vocab_en=None, cut=False):
        self.text_de = []
        with open(source_file_de) as file:
            for line in file.readlines():
                self.text_de.append(np.array(line.split(' ')))
        self.text_de = np.array(self.text_de, dtype=type(self.text_de[0]))
        if cut:
            new_ind = np.random.choice(self.text_de.shape[0],
                                       self.text_de.shape[0] // 2)
            self.text_de = self.text_de[new_ind]
        self.text_en = None

        self.specials = ['<pad>', '<bos>', '<eos>', '<unk>']

        self.vocab_de = vocab_de
        self.itos_de = self.vocab_de.get_itos()
        self.vocab_en = None
        self.itos_en = None

        self.pad_index = self.vocab_de['<pad>']
        self.bos_index = self.vocab_de['<bos>']
        self.eos_index = self.vocab_de['<eos>']
        self.unk_index = self.vocab_de['<unk>']
        self.max_len = max_len

        if source_file_en is not None:
            self.text_en = []
            with open(source_file_en) as file:
                for line in file.readlines():
                    self.text_en.append(np.array(line.split(' ')))
            self.text_en = np.array(self.text_en, dtype=type(self.text_en[0]))
            if cut:
                self.text_en = self.text_en[new_ind]
            self.vocab_en = vocab_en
            self.itos_en = self.vocab_en.get_itos()

    def __len__(self):
        return len(self.text_de)

    def __getitem__(self, item):
        encoded_de = self.encode(self.text_de[item])

        if self.text_en is not None:
            encoded_en = self.encode(self.text_en[item], lng='en')
            return encoded_de, encoded_en

        return encoded_de

    def str_to_idx(self, text, lng='de'):
        if lng == 'de':
            return [self.vocab_de[word] for word in text]
        return [self.vocab_en[word] for word in text]

    def idx_to_str(self, idx, lng='de'):
        if lng == 'de':
            return [self.itos_de[index] for index in idx]
        return [self.itos_en[index] for index in idx]

    def encode(self, chars, lng='de'):
        chars = ['<bos>'] + list(chars) + ['<eos>']
        return self.str_to_idx(chars, lng)

    def decode(self, idx, lng='de'):
        chars = self.idx_to_str(idx, lng)
        return ' '.join(char for char in chars if char not in self.specials)

    def get_last_word_in_str(self, item):
        return self.text_de[item][-1]
