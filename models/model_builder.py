from abc import ABC

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_model(layers=None, in_2d_channels=None, num_class=None, dropout=None, lr=None, trans=None, mode=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = CNN1dT2d(in_2d_channels=in_2d_channels,

                     layer_list=layers,

                     dropout=dropout,

                     num_class=num_class,

                     trans=trans,

                     mode=mode).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    return model, criterion, optimizer, device


class CNN1dT2d(nn.Module, ABC):

    def __init__(self, in_2d_channels, dropout=0.5, layer_list=None, num_class=None, trans='GC', mode=None):
        super(CNN1dT2d, self).__init__()
        self.dropout = dropout
        if mode == 'LSTM':
            self.net_1d = LSTM(in_size=1, hidden_size=16, out_size=8, num_layers=9)
        elif mode == 'GRU':
            self.net_1d = GRU(in_size=1, hidden_size=16, out_size=8, num_layers=11)
        elif mode == 'Transformer':
            self.net_1d = Transformer()
        else:
            self.net_1d = CNN1D(in_channels=1, layer_list=layer_list)

        if trans == 'RG':
            self.translation = RGTrans()
        elif trans == 'GT':
            self.translation = GTTrans()
        elif trans == 'GC':
            self.translation = GCTrans()
        elif trans == 'GG':
            self.translation = GGTrans()
        else:
            raise RuntimeError('Not support this {} trans'.format(trans))

        self.net_2d = CNN2D(in_channels=in_2d_channels)

        self.fc = nn.Sequential(nn.Linear(64, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(1024, num_class))

    def forward(self, x):

        x_1d_in = x.reshape(-1, x.size(-1)).unsqueeze(1)
        x_1d_out = self.net_1d(x_1d_in)

        x_trans_in = x_1d_out.reshape((-1, x.size(1)) + x_1d_out.size()[-2:])

        x_2d_input = self.translation(x_trans_in)
        x_2d_out = self.net_2d(x_2d_input)
        out = self.fc(x_2d_out)

        return out


class RGTrans(nn.Module, ABC):

    def __init__(self):
        super(RGTrans, self).__init__()
        self.trans = nn.Sequential(nn.Linear(8, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 1))

    def forward(self, x):

        out = torch.repeat_interleave(x.unsqueeze(-1), x.size(-1), dim=-1)
        out = out - out.permute(0, 1, 2, 4, 3)
        out = out.permute(0, 1, 3, 4, 2).reshape(-1, x.size(-2))
        out = self.trans(out)
        out = out.reshape(x.size()[:2] + (25, 25))

        return out


class GGTrans(nn.Module, ABC):

    def __init__(self):
        super(GGTrans, self).__init__()
        self.trans = nn.Sequential(nn.Linear(200, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 625))

    def forward(self, x):
        out = x.reshape((-1, ) + x.size()[-2:])
        out = x.reshape(out.size(0), -1)
        out = self.trans(out)
        out = out.reshape(x.size()[:2] + (25, 25))

        return out


class GTTrans(nn.Module, ABC):

    def __init__(self):
        super(GTTrans, self).__init__()
        self.trans = nn.Conv1d(8, 25, stride=1, kernel_size=1, bias=False)

    def forward(self, x):

        out = x.reshape((-1, ) + x.size()[-2:])
        out = self.trans(out)
        out = out.reshape(x.size()[:2] + out.size()[-2:])
        out = torch.matmul(out.permute(0, 1, 3, 2), out)

        return out


class GCTrans(nn.Module, ABC):

    def __init__(self):
        super(GCTrans, self).__init__()
        self.trans = nn.Conv1d(8, 25, stride=1, kernel_size=1, bias=False, padding=1)

    def forward(self, x):
        out = x.reshape((-1, ) + x.size()[-2:])
        out = self.trans(out)
        out = out.reshape(x.size()[:2] + out.size()[-2:])
        out = torch.matmul(out, out.permute(0, 1, 3, 2))

        return out


class Block(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, stride=1, down=None):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.down = down

    def forward(self, x):

        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down is not None:
            res = self.down(x)

        out = out + res
        out = self.relu(out)

        return out


class CNN1D(nn.Module, ABC):
    def __init__(self, in_channels, layer_list=None):
        super(CNN1D, self).__init__()
        self.channel_list = [8, 32, 16, 8]
        self.layer1 = self._make_layer(Block, in_channels, self.channel_list[0], layer_list[0],
                                       stride=2, max_pool=True)
        self.layer2 = self._make_layer(Block, self.channel_list[0], self.channel_list[1], layer_list[1],
                                       stride=2)
        self.layer3 = self._make_layer(Block, self.channel_list[1], self.channel_list[2], layer_list[2],
                                       stride=1)
        self.layer4 = self._make_layer(Block, self.channel_list[2], self.channel_list[3], layer_list[3],
                                       stride=1)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, blocks, stride=1, max_pool=None, down=None):

        if in_channels != out_channels:
            down = nn.Conv1d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False)
        block_list = [block(in_channels, out_channels, stride, down)]

        for i in range(1, blocks):
            block_list.append(block(out_channels, out_channels))

        if max_pool:
            max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            block_list.append(max_pool)

        return nn.Sequential(*block_list)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class CNN2D(nn.Module, ABC):
    def __init__(self, in_channels):
        super(CNN2D, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2,  padding=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.avg2d = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_0 = self.conv0(x)
        x_m = self.max_pool(x_0)

        x_1 = self.conv1(x_m)

        x_2 = self.conv2(x_1)

        x_3 = self.conv3(x_2)

        x_4 = self.conv4(x_3)

        x_out = self.avg2d(x_4).squeeze(-1).squeeze(-1)
        return x_out


class LSTM(nn.Module, ABC):
    def __init__(self, in_size, hidden_size, num_layers=None, out_size=None):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x


class GRU(nn.Module, ABC):
    def __init__(self, in_size, hidden_size, num_layers=None, out_size=None):
        super(GRU, self).__init__()
        self.lstm = nn.GRU(input_size=in_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x


class Transformer(nn.Module, ABC):
    def __init__(self, d_model=8, d_inner=32,
                 n_layers=3, n_head=6, d_k=32, d_v=32, dropout=0.1, n_position=25):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_position=n_position, d_model=d_model,
                               d_inner=d_inner, n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

    def forward(self, x):

        output = self.encoder(x)

        return output.permute(0, 2, 1)


class Encoder(nn.Module, ABC):

    def __init__(
            self, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=8, d_inner=8, dropout=0.1, n_position=25):

        super().__init__()

        self.emb = nn.Linear(1, 8)
        self.position_enc = PositionalEncoding(8, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq):
        src_seq = src_seq.permute(0, 2, 1)
        enc_output = self.emb(src_seq)
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output


class PositionalEncoding(nn.Module, ABC):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_hid):

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        pos = self.pos_table[:, :x.size(1)]
        pos = pos.clone()
        pos = pos.detach()
        return x + pos


class EncoderLayer(nn.Module, ABC):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class MultiHeadAttention(nn.Module, ABC):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class ScaledDotProductAttention(nn.Module, ABC):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


class PositionWiseFeedForward(nn.Module, ABC):

    def __init__(self, d_in=8, d_hid=32, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
