import sys

sys.path.append('/apdcephfs/private_liamlwang/KWS_Text/')
sys.path.append('/apdcephfs/private_liamlwang/KWS_Text/models')
sys.path.append('/apdcephfs/private_liamlwang/KWS_Text/utils')
import torch
import torchaudio
import numpy as np
from config import config
from random import choice
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class SpeechDatasetV2(Dataset):
    def __init__(self, set='train', mode=''):
        self.commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                         'backward', 'forward', 'follow', 'learn']
        self.unknow_words = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow", 'visual',
                             'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        self.all_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                          'backward', 'forward', 'follow', 'learn', "bed", "bird", "cat", "dog", "happy", "house",
                          "marvin", "sheila", "tree", "wow", 'visual',
                          'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        self.train_file = config.TRAIN.TRAINFILE
        self.valid_file = config.TRAIN.VALIDFILE
        self.test_file = config.TEST.TESTFILE
        self.data = []
        self.num_classes = len(self.commands) + 1
        if set == 'train':
            self.dataset_path = config.TRAIN.DATASETPATH
            self.file = self.train_file
        elif set == 'valid':
            self.dataset_path = config.TRAIN.DATASETPATH
            self.file = self.valid_file
        elif set == 'test':
            self.dataset_path = config.TEST.DATASETPATH
            self.file = self.test_file
        else:
            print('SpeechDatasetV2 set error!')
        with open(self.file, 'r') as f:
            for line in f.readlines():
                self.data.append(line.strip('\n').split('/', 1))
        self.x_data = [i[1] for i in self.data]
        self.y_data = [i[0] for i in self.data]
        self.num_data = len(self.y_data)
        self.mode = mode
        if self.mode != 'Text':
            self.text_emb = np.load(config.TEXTEMB, allow_pickle=True).item()

    def __getitem__(self, i):
        if self.mode == 'NoText': # return normal data: waveform, label
            waveform, _ = torchaudio.load(self.dataset_path + self.y_data[i] + '/' + self.x_data[i])
            waveform = self.normalize(waveform)
            label_num = self.class_to_num(self.y_data[i])
            return waveform, label_num
        elif self.mode == 'Text':
            waveform, _ = torchaudio.load(self.dataset_path + self.y_data[i] + '/' + self.x_data[i])
            waveform = self.normalize(waveform)
            label_num = self.class_to_num(self.y_data[i])
            match_word = self.y_data[i]
            unmatch_word = choice(self.all_words)
            while unmatch_word == match_word:
                unmatch_word = choice(self.all_words)
            match_word_vec = self.text_emb[match_word]
            unmatch_word_vec = self.text_emb[unmatch_word]
            return waveform, match_word_vec, unmatch_word_vec, label_num
        elif self.mode == 'TextAnchor':
            pos_waveform, _ = torchaudio.load(self.dataset_path + self.y_data[i] + '/' + self.x_data[i])
            pos_waveform = self.normalize(pos_waveform)
            pos_label_num = self.class_to_num(self.y_data[i])
            neg_index = choice(range(self.num_data))
            while self.y_data[i] == self.y_data[neg_index]:
                neg_index = choice(range(self.num_data))
            neg_waveform, _ = torchaudio.load(self.dataset_path + self.y_data[neg_index] + '/' + self.x_data[neg_index])
            neg_waveform = self.normalize(neg_waveform)
            neg_label_num = self.class_to_num(self.y_data[neg_index])
            pos_word_vec = self.text_emb[self.y_data[i]]
            return pos_waveform, neg_waveform, pos_word_vec, pos_label_num, neg_label_num
        elif self.mode == 'CCA':
            waveform, _ = torchaudio.load(self.dataset_path + self.y_data[i] + '/' + self.x_data[i])
            waveform = self.normalize(waveform)
            label_num = self.class_to_num(self.y_data[i])
            word = self.y_data[i]
            word_vec = self.text_emb[word]
            return waveform, word_vec, label_num
        else:
            print('[ERROR] Wrong MODE!')
            exit(1)

    def __len__(self):
        return len(self.data)

    def class_to_num(self, word):
        if word in self.commands:
            label_num = self.commands.index(word)
        if word in self.unknow_words:
            label_num = self.num_classes - 1
        return label_num

    def normalize(self, tensor):
        # Subtract the mean, and scale to the interval [-1,1]
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean / tensor_minusmean.abs().max()



class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpeechDatasetV2('valid')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    for data in dataloader:
        # start = time.time()
        waveform, labels = data
        waveform = waveform.to(device)
        labels = labels.to(device)
        # end = time.time()
        # print('Running time: %s Seconds' % (end - start))
        print(waveform.size())
        # print(labels.size())
        # print(labels)
