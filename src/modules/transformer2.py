import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb


VALID_ACTIVATION = [a for a in dir(nn.modules.activation)
                    if not a.startswith('__')
                    and a not in ['torch', 'warnings', 'F', 'Parameter', 'Module']]
VALID_BATCHNORM_DIM = {1, 2, 3}


class DepthwiseSeperableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeperableConv1d, self).__init__()

        self.depthwise_conv1d = nn.Conv1d(
            in_channels, in_channels, kernel_size, groups=in_channels,
            padding=kernel_size // 2)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv1d(x)
        x = self.pointwise_conv1d(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, n_heads, max_pos_distance):
        super(MultiHeadSelfAttention, self).__init__()

        if output_size % n_heads != 0:
            raise ValueError(
                'MultiHeadSelfAttention: output_size({output_size}) isn\'t'
                'a multiplier of n_heads({n_heads})')

        self.output_size = output_size
        self.n_heads = n_heads
        self.d_head = output_size // n_heads
        self.max_pos_distance = max_pos_distance

        self.pos_embedding_K = nn.Embedding(2 * max_pos_distance + 1, self.d_head)
        self.pos_embedding_V = nn.Embedding(2 * max_pos_distance + 1, self.d_head)

        self.linears = nn.ModuleList([
            nn.Linear(input_size, output_size),
            nn.Linear(input_size, output_size),
            nn.Linear(input_size, output_size),
            nn.Linear(output_size, output_size)
        ])

    def forward(self, x, pad_mask):
        batch_size, input_len, *_ = x.shape

        Q, K, V = [l(x) for l in self.linears[:3]]
        Q, K, V = [
            x.reshape(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
            for x in (Q, K, V)
        ]

        pos_index = torch.arange(input_len).reshape(1, -1).repeat(input_len, 1)
        pos_index = pos_index - pos_index.t()
        pos_index = pos_index.clamp(-self.max_pos_distance, self.max_pos_distance)
        pos_index += self.max_pos_distance
        # pos_index = pos_index.to(dtype=torch.int64, device=constants.DEVICE)
        pos_index = pos_index.to(dtype=torch.int64, device=torch.device('cuda:0'))

        # calculate attention score (relative position representation #1)
        S1 = torch.matmul(Q, K.transpose(-1, -2))
        Q = Q.reshape(-1, input_len, self.d_head).transpose(0, 1)
        pos_emb_K = self.pos_embedding_K(pos_index)
        S2 = torch.matmul(Q, pos_emb_K.transpose(-1, -2)).transpose(0, 1)
        S2 = S2.reshape(batch_size, self.n_heads, input_len, input_len)
        S = (S1 + S2) / np.sqrt(self.d_head)

        # set score of V padding tokens to 0
        S *= pad_mask.to(dtype=torch.float32).reshape(batch_size, 1, 1, -1)
        A = F.softmax(S, dim=-1)

        # apply attention to get output (relative position representation #2)
        O1 = torch.matmul(A, V)
        A = A.reshape(-1, input_len, input_len).transpose(0, 1)
        pos_emb_V = self.pos_embedding_V(pos_index)
        O2 = torch.matmul(A, pos_emb_V).transpose(0, 1)
        O2 = O2.reshape(batch_size, self.n_heads, input_len, self.d_head)
        output = O1 + O2
        output = output.transpose(1, 2).reshape(batch_size, -1, self.output_size)
        output = self.linears[-1](output)

        return output


class PointwiseFeedForward(nn.Module):
    def __init__(self, input_size):
        super(PointwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(input_size, input_size)
        self.activation = Activation('ReLU')
        self.linear2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)

        return x


class Activation(nn.Module):
    def __init__(self, activation, *args, **kwargs):
        super(Activation, self).__init__()

        if activation in VALID_ACTIVATION:
            self.activation = \
                getattr(nn.modules.activation, activation)(*args, **kwargs)
        else:
            raise ValueError(
                'Activation: {activation} is not a valid activation function')

    def forward(self, x):
        return self.activation(x)


class BatchNormResidual(nn.Module):
    def __init__(self, sublayer, n_features, dim=1, transpose=False, activation=None,
                 dropout=0):
        super(BatchNormResidual, self).__init__()

        self.sublayer = sublayer
        if dim in VALID_BATCHNORM_DIM:
            batch_norm = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            self.batch_norm = batch_norm[dim - 1](n_features)
        else:
            raise ValueError(
                'BatchNormResidual: dim must be one of {{1, 2, 3}}, but got {dim}')
        self.transpose = transpose
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else None

    def forward(self, x, *args, **kwargs):
        y = self.sublayer(x, *args, **kwargs)

        if self.transpose:
            y = y.transpose(1, 2).contiguous()
        y = self.batch_norm(y)
        if self.transpose:
            y = y.transpose(1, 2)
        y += x

        if self.activation:
            y = self.activation(y)
        if self.dropout:
            y = self.dropout(y)

        return y


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_convs, kernel_size, n_heads,
                 max_pos_distance, dropout):
        super(EncoderBlock, self).__init__()

        self.convs = nn.ModuleList([
            BatchNormResidual(
                DepthwiseSeperableConv1d(d_model, d_model, kernel_size), d_model,
                activation=Activation('ReLU'), dropout=dropout)
            for _ in range(n_convs - 1)
        ])
        self.attention = BatchNormResidual(
            MultiHeadSelfAttention(d_model, d_model, n_heads, max_pos_distance),
            d_model, transpose=True, dropout=dropout)
        self.feedforward = BatchNormResidual(
            PointwiseFeedForward(d_model), d_model, transpose=True,
            activation=Activation('ReLU'), dropout=dropout)

    def forward(self, x, x_pad_mask):
        x = self.attention(x, x_pad_mask)
        x = self.feedforward(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size, n_heads, dropout, ph, n_convs=1, kernel_size=1, d_model=300, 
                 n_blocks=1, max_pos_distance=20):
        super(Encoder, self).__init__()

        self.conv = DepthwiseSeperableConv1d(input_size, d_model, kernel_size)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Activation('ReLU')
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else None
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model, n_convs, kernel_size, n_heads, max_pos_distance, dropout)
            for i in range(n_blocks)
        ])

    def forward(self, x, lens):
        x_pad_mask = torch.zeros_like(x[:, :, :1])
        for i, ll in enumerate(lens):
            x_pad_mask[i, :ll] = 1

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, x_pad_mask)

        return x
