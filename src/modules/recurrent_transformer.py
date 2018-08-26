import torch
import pdb
# from .transformer import EncoderLayer as TransformerEncoder
from .transformer2 import Encoder as TransformerEncoder
from .transformer2 import Connection, Seq2Vec
from .common import pad_seqs, BatchInnerProduct


class RecurrentTransformer(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, n_heads, dropout_rate, dim_ff,
                 dim_encoder=102, dim_encoder_ff=256):
        super(RecurrentTransformer, self).__init__()
        self.transformer = RecurrentTransformerEncoder(
            dim_embeddings,
            n_heads, dropout_rate, dim_ff,
            dim_encoder, dim_encoder_ff)
        self.last_encoder = TransformerEncoder(
            dim_encoder, dim_encoder, n_heads,
            dropout_rate, dim_encoder_ff
        )
        self.attn = Connection(dim_encoder, dim_encoder, n_heads,
                               dropout_rate, dim_encoder_ff)
        self.seq2vec = Seq2Vec(dim_encoder, dim_encoder, n_heads, dropout_rate)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_encoder, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.similarity = BatchInnerProduct()
        self.register_buffer('padding', torch.zeros(dim_embeddings))
        self.register_buffer('padding2', torch.zeros(dim_encoder))

    def forward(self, context, context_ends, options, option_lens):
        batch_size = context.size(0)
        context = self.transformer(context, context_ends)
        context_lens, context = pad_seqs(context, self.padding2)
        context_vecs = self.seq2vec(context, context_lens)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option = self.transformer.encoder(
            #     option,
            #     [option_lens[b][i] for b in range(batch_size)])

            opt_lens = [option_lens[b][i]
                        for b in range(batch_size)]

            option = self.transformer.encoder(option[:, :max(opt_lens)],
                                              opt_lens)
            attn_co = self.attn(context, option, context_lens)
            attn_oc = self.attn(option, context, opt_lens)

            ctx_vecs = self.seq2vec(attn_oc, opt_lens)
            opt_vecs = self.seq2vec(attn_co, context_lens)

            # logit = self.mlp(vecs).squeeze(-1)
            logit = self.similarity(ctx_vecs, opt_vecs)
            logits.append(logit)

        # pdb.set_trace()
        logits = torch.stack(logits, 1)
        return logits


class RecurrentTransformerEncoder(torch.nn.Module):
    """ 

    Args:

    """
    def __init__(self, dim_embeddings,
                 n_heads=6, dropout_rate=0.1, dim_ff=128,
                 dim_encoder=102, dim_encoder_ff=256):
        super(RecurrentTransformerEncoder, self).__init__()
        self.encoder = TransformerEncoder(
            dim_embeddings,
            dim_encoder,
            n_heads,
            dropout_rate,
            dim_encoder_ff
        )
        self.connection = Connection(
            dim_encoder,
            dim_encoder,
            n_heads,
            dropout_rate,
            dim_ff)
        self.register_buffer('padding', torch.zeros(dim_embeddings))
        self.register_buffer('padding2', torch.zeros(dim_encoder))

    def forward(self, seqs, ends):
        first_ends = [min(end[0], 50) for end in ends]
        encoded = self.encoder(seqs[:, :max(first_ends)], first_ends)
        encoded = [seq[:end] for seq, end in zip(encoded, first_ends)]

        context_lens = list(map(len, ends))
        batch_size = seqs.size(0)

        for i in range(0, max(context_lens) - 1):
            workings = list(
                filter(lambda j: context_lens[j] > i + 1,
                       range(batch_size))
            )

            currs = []
            prevs = []
            for working in workings:
                start, end = ends[working][i], ends[working][i + 1]
                if i == 0:
                    curr_len = min(end - start, 150)
                else:
                    curr_len = min(end - start, 100)

                curr = seqs[working, start:start + curr_len]
                currs.append(curr)
                prevs.append(encoded[working])

            curr_lens, currs = pad_seqs(currs, self.padding)
            prev_lens, prevs = pad_seqs(prevs, self.padding2)

            currs = self.encoder(currs, curr_lens)
            outputs = self.connection(prevs, currs, prev_lens)

            for j, working in enumerate(workings):
                start, end = ends[working][i], ends[working][i + 1]
                seq_len = min(end - start, 50)
                encoded[working] = outputs[j][-seq_len:]

        return encoded


