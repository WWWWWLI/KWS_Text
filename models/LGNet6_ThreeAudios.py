import torch.nn as nn
import torch
from models.attention_model import MultiHeadAttention, PositionalEncoding
from models.initmodel import normal_init, constant_init
import torch.nn.functional as F
import torchaudio
from config import config


class LGNet6_ThreeAudios(nn.Module):
    def __init__(self, k=1.5, c_in=40):
        super(LGNet6_ThreeAudios, self).__init__()
        self.torchmfcc = torchaudio.transforms.MFCC(n_mfcc=c_in,
                                                    melkwargs={'win_length': 400, 'hop_length': 160, 'pad': 0})
        self.conv1 = nn.Conv2d(40, int(24 * k), (3, 1), padding=(1, 0))
        self.LGBlock1 = LGBlock(int(24 * k), int(32 * k), 2, 51, int(32 * k))
        self.LGBlock2 = LGBlock(int(32 * k), int(32 * k), 1, 51, int(32 * k))
        self.LGBlock3 = LGBlock(int(32 * k), int(48 * k), 2, 26, int(48 * k))
        self.LGBlock4 = LGBlock(int(48 * k), int(48 * k), 1, 26, int(48 * k))
        self.LGBlock5 = LGBlock(int(48 * k), int(56 * k), 2, 13, int(56 * k))
        self.LGBlock6 = LGBlock(int(56 * k), int(56 * k), 1, 13, int(56 * k))
        self.pool = nn.AvgPool2d((13, 1))
        self.classifier = nn.Linear(int(56 * k), config.TEXT_PROJ_DIM)
        self.classifier2 = nn.Linear(config.TEXT_PROJ_DIM, config.NumClasses)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def forward(self, anchor_waveform, pos_waveform, neg_waveform):
        anchor_waveform = self.torchmfcc(anchor_waveform)
        anchor_waveform = torch.transpose(anchor_waveform, 1, 2)
        anchor_waveform = torch.transpose(anchor_waveform, 2, 3)
        anchor_waveform = self.conv1(anchor_waveform)
        anchor_waveform = self.LGBlock1(anchor_waveform)  # (16,32,t,1)
        anchor_waveform = self.LGBlock2(anchor_waveform)  # (16,12,t,1)
        anchor_waveform = self.LGBlock3(anchor_waveform)
        anchor_waveform = self.LGBlock4(anchor_waveform)
        anchor_waveform = self.LGBlock5(anchor_waveform)
        anchor_waveform = self.LGBlock6(anchor_waveform)
        anchor_waveform = self.pool(anchor_waveform)
        anchor_waveform = anchor_waveform.contiguous().view(anchor_waveform.size(0), -1)
        anchor_waveform = self.dropout(anchor_waveform)
        anchor_waveform = self.classifier(anchor_waveform)
        audio_embedding_anchor = anchor_waveform
        anchor_waveform = self.dropout(anchor_waveform)
        anchor_output = self.classifier2(anchor_waveform)

        pos_waveform = self.torchmfcc(pos_waveform)
        pos_waveform = torch.transpose(pos_waveform, 1, 2)
        pos_waveform = torch.transpose(pos_waveform, 2, 3)
        pos_waveform = self.conv1(pos_waveform)
        pos_waveform = self.LGBlock1(pos_waveform)  # (16,32,t,1)
        pos_waveform = self.LGBlock2(pos_waveform)  # (16,12,t,1)
        pos_waveform = self.LGBlock3(pos_waveform)
        pos_waveform = self.LGBlock4(pos_waveform)
        pos_waveform = self.LGBlock5(pos_waveform)
        pos_waveform = self.LGBlock6(pos_waveform)
        pos_waveform = self.pool(pos_waveform)
        pos_waveform = pos_waveform.contiguous().view(pos_waveform.size(0), -1)
        pos_waveform = self.dropout(pos_waveform)
        pos_waveform = self.classifier(pos_waveform)
        audio_embedding_pos = pos_waveform

        neg_waveform = self.torchmfcc(neg_waveform)
        neg_waveform = torch.transpose(neg_waveform, 1, 2)
        neg_waveform = torch.transpose(neg_waveform, 2, 3)
        neg_waveform = self.conv1(neg_waveform)
        neg_waveform = self.LGBlock1(neg_waveform)  # (16,32,t,1)
        neg_waveform = self.LGBlock2(neg_waveform)  # (16,12,t,1)
        neg_waveform = self.LGBlock3(neg_waveform)
        neg_waveform = self.LGBlock4(neg_waveform)
        neg_waveform = self.LGBlock5(neg_waveform)
        neg_waveform = self.LGBlock6(neg_waveform)
        neg_waveform = self.pool(neg_waveform)
        neg_waveform = neg_waveform.view(neg_waveform.size(0), -1)
        neg_waveform = self.dropout(neg_waveform)
        neg_waveform = self.classifier(neg_waveform)
        audio_embedding_neg = neg_waveform

        return anchor_output, audio_embedding_anchor, audio_embedding_pos, audio_embedding_neg

    def init_weights(self):
        print('[Message] Init models.')
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


class LGBlock(nn.Module):
    def __init__(self, c_in, c_out, s, max_len, output_dim):
        super(LGBlock, self).__init__()
        self.TConv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 1), stride=s, bias=False, padding=(1, 0)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(3, 1), stride=1, bias=False, padding=(1, 0)),
            nn.BatchNorm2d(c_out),
        )

        self.shortcut = nn.Sequential()
        if s != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), stride=s, bias=False),
                nn.BatchNorm2d(c_out),
            )

        self.pe = PositionalEncoding(d_model=c_out, dropout=0.1, max_len=max_len)
        self.mh = MultiHeadAttention(model_dim=c_out, num_heads=1, output_dim=output_dim, dropout=0.5,
                                     share_weight=False)

    def forward(self, x):
        # (16,1,t,c_in)
        res = self.shortcut(x)  # (16,c_out,t',1)
        x = self.TConv(x)  # (16,c_out,t',1)
        x += res
        x = F.relu(x)

        x = torch.squeeze(x)  # (16,c_out,t')
        x = torch.transpose(x, 1, 2)  # (16,t',c_out)
        x = self.pe(x)  # (16,t',c_out)
        x, _, _ = self.mh(x, x, x)  # (16,t',c_out)
        x = torch.transpose(x, 1, 2)  # (16,c_out,t')
        x = torch.unsqueeze(x, 3)  # (16,c_out,t',1)

        return x


if __name__ == '__main__':
    from torchsummaryX import summary


    def print_net(net, device=None):
        anchor_waveform = torch.randn(16, 1, 16000)
        pos_waveform = torch.randn(16, 1, 16000)
        neg_waveform = torch.randn(16, 1, 16000)
        summary(net.to(device), anchor_waveform.to(device), pos_waveform.to(device), neg_waveform.to(device))


    model = LGNet6_ThreeAudios()
    print_net(model)
