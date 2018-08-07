import math
import torch
import pdb
from .transformer import EncoderLayer as TransformerEncoder
# from .transformer2 import Encoder as TransformerEncoder


class DualRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product'):
        super(DualRNN, self).__init__()
        self.context_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]

    def forward(self, context, context_lens, options, option_lens):
        context_hidden = self.context_encoder(context, context_lens)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class HierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product'):
        super(HierRNN, self).__init__()
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]

    def forward(self, context, context_ends, options, option_lens):
        context_hidden = self.context_encoder(context, context_ends)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class RecurrentTransformer(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, n_heads, dropout_rate, dim_ff):
        super(RecurrentTransformer, self).__init__()
        self.transformer = RecurrentTransformerEncoder(
            dim_embeddings,
            n_heads, dropout_rate, dim_ff)
        self.last_encoder = TransformerEncoder(dim_embeddings, n_heads,
                                               dropout_rate, dim_ff)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_embeddings, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.register_buffer('padding', torch.zeros(dim_embeddings))

    def forward(self, context, context_ends, options, option_lens):
        batch_size = context.size(0)
        encoded = self.transformer(context, context_ends)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option = self.transformer.encoder(
                option,
                [option_lens[b][i] for b in range(batch_size)])
            option = [opt[:ol]
                      for opt, ol in zip(option,
                                         [option_lens[b][i]
                                          for b in range(batch_size)])]

            tr_inputs = [torch.cat([encoded[b], opt], 0)
                         for b, (opt, ol) in enumerate(zip(option,
                                                           option_lens))]
            lens, tr_inputs = pad_seqs(tr_inputs, self.padding)

            outputs = self.transformer.recurrent(
                tr_inputs, lens
            )

            # logit.size() == (batch,)
            logit = self.last_encoder(outputs, lens)
            logit = logit.max(1)[0]
            logit = self.mlp(logit).squeeze(-1)
            logits.append(logit)

        logits = torch.stack(logits, 1)
        return logits


class HierRNNEncoder(torch.nn.Module):
    """ 

    Args:

    """
    def __init__(self, dim_embeddings, dim_hidden1=128, dim_hidden2=128):
        super(HierRNNEncoder, self).__init__()
        self.rnn1 = torch.nn.LSTM(dim_embeddings,
                                  dim_hidden1,
                                  1,
                                  bidirectional=True,
                                  batch_first=True)
        self.rnn2 = LSTMEncoder(2 * dim_hidden1, dim_hidden2)
        self.register_buffer('padding', torch.zeros(2 * dim_hidden2))

    def forward(self, seq, ends):
        batch_size = seq.size(0)

        outputs, _ = self.rnn1(seq)
        end_outputs = [[outputs[b, end] for end in ends[b]]
                       for b in range(batch_size)]
        lens, end_outputs = pad_and_cat(end_outputs, self.padding)
        encoded = self.rnn2(end_outputs, list(map(len, ends)))
        return encoded


class RecurrentTransformerEncoder(torch.nn.Module):
    """ 

    Args:

    """
    def __init__(self, dim_embeddings,
                 n_heads=6, dropout_rate=0.1, dim_ff=128):
        super(RecurrentTransformerEncoder, self).__init__()
        self.encoder = TransformerEncoder(
            dim_embeddings,
            n_heads,
            dropout_rate,
            dim_ff
        )
        self.recurrent = TransformerEncoder(
            dim_embeddings,
            n_heads,
            dropout_rate,
            dim_ff)
        self.register_buffer('padding', torch.zeros(dim_embeddings))

    def forward(self, seqs, ends):
        first_ends = [end[0] + 1 for end in ends]
        encoded = [seq for seq in self.encoder(seqs[:, :max(first_ends)],
                                               first_ends)]
        context_lens = list(map(len, ends))
        batch_size = seqs.size(0)

        for i in range(0, max(context_lens) - 1):
            workings = filter(lambda j: context_lens[j] > i + 1,
                              range(batch_size))
            tr_inputs = []
            for working in workings:
                start, end = ends[working][i], ends[working][i + 1]
                tr_input = torch.cat(
                    [encoded[working], seqs[working, start:end]], 0
                )
                tr_inputs.append(tr_input)

            lens, tr_inputs = pad_seqs(tr_inputs, self.padding)
            outputs = self.recurrent(tr_inputs, lens)

            for j, working in enumerate(workings):
                encoded[working] = outputs[j][- end + start:]

        return encoded


class BatchInnerProduct(torch.nn.Module):
    """ 

    Args:

    """

    def __init__(self):
        super(BatchInnerProduct, self).__init__()

    def forward(self, a, b):
        return (a * b).sum(-1)


class LSTMEncoder(torch.nn.Module):
    """ 

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden):
        super(LSTMEncoder, self).__init__()
        self.rnn = torch.nn.LSTM(dim_embeddings,
                                 dim_hidden,
                                 1,
                                 bidirectional=True,
                                 batch_first=True)

    def forward(self, seqs, seq_lens):
        _, hidden = self.rnn(seqs)
        _, c = hidden
        return c.transpose(1, 0).contiguous().view(c.size(1), -1)


class RankLoss(torch.nn.Module):
    """ 
    Args:

    """

    def __init__(self, margin=0):
        super(RankLoss, self).__init__()
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, logits, labels):
        positive_mask = (1 - labels).byte()
        positive_min = torch.min(logits.masked_fill(
            positive_mask, math.inf), -1)[0]

        negative_mask = labels.byte()
        negative_max = torch.max(logits.masked_fill(
            negative_mask, -math.inf), -1)[0]

        loss = self.margin_ranking_loss(positive_min, negative_max,
                                        torch.ones_like(negative_max))
        loss = (negative_max - positive_min).mean()
        return loss


class NLLLoss(torch.nn.Module):
    """ 
    Args:

    """

    def __init__(self, epsilon=1e-6):
        super(NLLLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        batch_size = logits.size(0)
        probs = torch.nn.functional.softmax(logits, -1)
        loss = (-torch.log(probs + self.epsilon) *
                labels.float()).sum() / batch_size
        return loss


def pad_and_cat(tensors, padding):
    """ Pad lists to have same number of elements, and concatenate
    those elements to a 3d tensor.

    Args:
        tensors (list of list of Tensors): Each list contains
            list of operand embeddings. Each operand embedding is of
            size (dim_element,).
        padding (Tensor):
            Element used to pad lists, with size (dim_element,).

    Return:
        n_tensors (list of int): Length of lists in tensors.
        tensors (Tensor): Concatenated tensor after padding the list.
    """
    n_tensors = [len(ts) for ts in tensors]
    pad_size = max(n_tensors)

    # pad to has same number of operands for each problem
    tensors = [ts + (pad_size - len(ts)) * [padding]
               for ts in tensors]

    # tensors.size() = (batch_size, pad_size, dim_hidden)
    tensors = torch.stack([torch.stack(t)
                           for t in tensors], dim=0)

    return n_tensors, tensors


def pad_seqs(tensors, pad_element):
    lengths = list(map(len, tensors))
    max_length = max(lengths)
    padded = []
    for tensor, length in zip(tensors, lengths):
        if max_length > length:
            padding = torch.stack(
                [pad_element] * (max_length - length), 0
            )
            padded.append(torch.cat([tensor, padding], 0))
        else:
            padded.append(tensor)

    return lengths, torch.stack(padded, 0)
