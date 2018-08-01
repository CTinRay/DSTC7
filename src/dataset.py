import pdb
import random
import torch
from torch.utils.data import Dataset


class DSTC7Dataset(Dataset):
    def __init__(self, data, padding=0,
                 n_negative=4, n_positive=1,
                 context_padded_len=400, option_padded_len=50):
        self.data = data
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.context_padded_len = context_padded_len
        self.option_padded_len = option_padded_len
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])

        # sample positive indices
        positive_indices = list(range(data['n_corrects']))
        random.shuffle(positive_indices)
        positive_indices = positive_indices[:self.n_positive]

        # sample negative indices
        negative_indices = list(range(data['n_corrects'],
                                      len(data['options'])))
        random.shuffle(negative_indices)
        negative_indices = negative_indices[:self.n_negative]
        
        data['options'] = (
            [data['options'][i] for i in positive_indices]
            + [data['options'][i] for i in negative_indices]
        )

        data['labels'] = [1] * self.n_positive + [0] * self.n_negative

        return data

    def collate_fn(self, datas):
        batch = {}

        # collate lists
        batch['id'] = [data['id'] for data in datas]
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['utterance_ends'] = [data['utterance_ends'] for data in datas]
        batch['labels'] = torch.tensor([data['labels'] for data in datas])

        # build tensor of context
        batch['context_lens'] = [len(data['context']) for data in datas]
        padded_len = min(self.context_padded_len, max(batch['context_lens']))
        batch['context'] = torch.tensor(
            [pad_to_len(data['context'], padded_len, self.padding)
             for data in datas]
        )

        # build tensor of options
        batch['option_lens'] = [[len(opt) for opt in data['options']]
                                for data in datas]
        batch['options'] = torch.tensor(
            [[pad_to_len(opt, self.option_padded_len, self.padding)
              for opt in data['options']]
             for data in datas]
        )

        return batch


def pad_to_len(arr, padded_len, padding=0):
    padded = [padding] * padded_len
    n_copy = min(len(arr), padded_len)
    padded[:n_copy] = arr[:n_copy]
    return padded