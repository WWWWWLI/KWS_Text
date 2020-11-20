import torch.nn as nn
import torch
from models.attention_model import MultiHeadAttention, PositionalEncoding
from models.initmodel import normal_init, constant_init
import torch.nn.functional as F
import torchaudio
from config import config


class LGNet6(nn.Module):
    def __init__(self, k=1.5, c_in=40):
        super(LGNet6, self).__init__()
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
        if config.TRAIN.MODE == 'FinetuneSilence':
            for p in self.parameters():
                p.requires_grad = False
            self.classifier = nn.Linear(int(56 * k), config.TEXT_PROJ_DIM)
            self.classifier2 = nn.Linear(config.TEXT_PROJ_DIM, config.NumClasses)

        else:
            self.classifier = nn.Linear(int(56 * k), config.NumClasses)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def forward(self, x):
        x = self.torchmfcc(x)
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 2, 3)
        x = self.conv1(x)
        x = self.LGBlock1(x)  # (16,32,t,1)
        x = self.LGBlock2(x)  # (16,12,t,1)
        x = self.LGBlock3(x)
        x = self.LGBlock4(x)
        x = self.LGBlock5(x)
        x = self.LGBlock6(x)
        x = self.pool(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        if config.TRAIN.MODE == 'FinetuneSilence':
            x = self.dropout(x)
            x = self.classifier2(x)
        return x

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
    import os
    from torchsummaryX import summary

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LGNet6().to(device)
    x = torch.zeros((16, 1, 16000)).to(device)

    summary(model, x)
