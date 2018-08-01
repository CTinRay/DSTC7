import re
import torch


class Embeddings:
    """
    Args:
        embedding_path (str): Path where embeddings are loaded from.
        words (None or list): If not None, only load embedding of the words in
            the list.
        oov_as_unk (bool): If argument `words` are provided, whether or not
            treat words in `words` but not in embedding file as `<unk>`. If
            true, OOV will be mapped to the index of `<unk>`. Otherwise,
            embeddings of those OOV will be randomly initialize and their
            indices will be after non-OOV.
    """

    def __init__(self, embedding_path, words=None, oov_as_unk=True):
        self._load_embeddings(embedding_path, set(words))

        if words is not None and not oov_as_unk:
            # initialize word vector for OOV
            for word in words:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)

            oov_embeddings = torch.nn.init.uniform_(
                torch.empty(len(self.word_dict) - self.embedding.size(0),
                            self.embedding.size(1)))

            self.embeddings = torch.cat([self.embedding, oov_embeddings], 0)

    def to_index(self, word):
        """
        word (str)

        Return:
             index of the word. If the word is not in `words` and not in the
             embedding file, then index of `<unk>` will be returned.
        """
        if word not in self.word_dict:
            return self.word_dict['<unk>']
        else:
            return self.word_dict[word]

    def get_dim(self):
        return self.embeddings.size(1)

    def get_vocabulary_size(self):
        return self.embeddings.size(0)

    def add(self, word, vector=None):
        if vector is not None:
            vector.view_(1, -1)
        else:
            vector = torch.empty(1, self.get_dim())
            torch.nn.init.uniform_(vector)
        self.embeddings = torch.cat([self.embeddings, vector], 0)
        self.word_dict[word] = len(self.word_dict)

    def _load_embeddings(self, embedding_path, words):
        word_dict = {}
        embedding = []

        with open(embedding_path) as fp:

            row1 = fp.readline()
            # if the first row is not header
            if not re.match('^[0-9]+ [0-9]+$', row1):
                # seek to 0
                fp.seek(0)
            # otherwise ignore the header

            for i, line in enumerate(fp):
                cols = line.rstrip().split(' ')
                word = cols[0]

                # skip word not in words if words are provided
                if words is not None and word not in words:
                    continue
                else:
                    word_dict[word] = len(word_dict)
                    embedding.append([float(v) for v in cols[1:]])

            if '</s>' not in word_dict:
                word_dict['</s>'] = len(word_dict)
                embedding.append([0] * len(embedding[0]))

            if '<unk>' not in word_dict:
                word_dict['<unk>'] = len(word_dict)
                embedding.append([0] * len(embedding[0]))

        self.word_dict = word_dict
        self.embeddings = torch.tensor(embedding)