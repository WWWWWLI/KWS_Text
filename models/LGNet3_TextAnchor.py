import torch.nn as nn
import torch
from models.attention_model import MultiHeadAttention, PositionalEncoding
from models.initmodel import normal_init, constant_init
import torch.nn.functional as F
import torchaudio
from config import config


class LGNet3_TextAnchor(nn.Module):
    def __init__(self, k=1, c_in=40):
        super(LGNet3_TextAnchor, self).__init__()
        self.torchmfcc = torchaudio.transforms.MFCC(n_mfcc=c_in,
                                                    melkwargs={'win_length': 400, 'hop_length': 160, 'pad': 0})
        self.conv1 = nn.Conv2d(40, int(24 * k), (3, 1), padding=(1, 0))
        self.LGBlock1 = LGBlock(int(24 * k), int(32 * k), 2, 51, int(32 * k))
        # self.LGBlock2 = LGBlock(int(32 * k), int(32 * k), 1, 51, int(32 * k))
        self.LGBlock2 = LGBlock(int(32 * k), int(48 * k), 2, 26, int(48 * k))
        # self.LGBlock3 = LGBlock(int(32 * k), int(48 * k), 2, 26, int(48 * k))
        self.LGBlock3 = LGBlock(int(48 * k), int(52 * k), 2, 13, int(52 * k))
        # self.pool = nn.AvgPool2d((26, 1))
        self.pool = nn.AvgPool2d((13, 1))

        # self.classifier = nn.Linear(int(48 * k), config.TEXT_PROJ_DIM)
        self.classifier = nn.Linear(int(52 * k), config.TEXT_PROJ_DIM)
        self.classifier2 = nn.Linear(config.TEXT_PROJ_DIM, config.NumClasses)
        self.dropout = nn.Dropout(0.5)

        self.match_text_proj = nn.Linear(config.TEXT_EMB_DIM, config.TEXT_PROJ_DIM)
        self.unmatch_text_proj = nn.Linear(config.TEXT_EMB_DIM, config.TEXT_PROJ_DIM)
        self.init_weights()

    def forward(self, pos, neg, anchor):
        pos = self.torchmfcc(pos)
        pos = torch.transpose(pos, 1, 2)
        pos = torch.transpose(pos, 2, 3)
        pos = self.conv1(pos)
        pos = self.LGBlock1(pos)  # (16,32,t,1)
        pos = self.LGBlock2(pos)  # (16,12,t,1)
        pos = self.LGBlock3(pos)
        pos = self.pool(pos)
        pos = pos.contiguous().view(pos.size(0), -1)
        pos = self.dropout(pos)
        pos = self.classifier(pos)
        audio_embedding_pos = pos
        pos = self.dropout(pos)
        pos = self.classifier2(pos)

        neg = self.torchmfcc(neg)
        neg = torch.transpose(neg, 1, 2)
        neg = torch.transpose(neg, 2, 3)
        neg = self.conv1(neg)
        neg = self.LGBlock1(neg)  # (16,32,t,1)
        neg = self.LGBlock2(neg)  # (16,12,t,1)
        neg = self.LGBlock3(neg)
        neg = self.pool(neg)
        neg = neg.contiguous().view(neg.size(0), -1)
        neg = self.dropout(neg)
        neg = self.classifier(neg)
        audio_embedding_neg = neg

        anchor = self.dropout(anchor)
        anchor = self.match_text_proj(anchor)

        return pos, audio_embedding_pos, audio_embedding_neg, anchor

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
            elif isinstance(m, nn.Conv1d):
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
        pos = torch.randn(16, 1, 16000)
        neg = torch.randn(16, 1, 16000)
        anchor = torch.randn(16, config.TEXT_EMB_DIM)
        summary(net.to(device), pos.to(device), neg.to(device), anchor.to(device))


    model = LGNet3_TextAnchor()
    print_net(model)
