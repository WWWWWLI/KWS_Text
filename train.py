# -*- coding: utf-8 -*-
########################################################################
#
#  Train Keyword Spotting Models
#
#  Author: Li Wang (1901213145@pku.edu.cn)
#
#  Date: 8 Sep, 2020
#
########################################################################


import torch
import torch.nn as nn
from torch import optim
from utils.dataloader2classes import SpeechDatasetV2
from torch.utils.data import DataLoader
from torchsummaryX import summary
import random
import numpy as np
import models
from config import config
from test import test_net
from datetime import datetime
import os
import shutil
import sys
from tqdm import tqdm
import time
from utils.cca_loss import cca_loss

sys.path.append(config.ROOTDIR + 'models')
sys.path.append(config.ROOTDIR + 'utils')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model():
    # create or load the model
    net = getattr(models, config.TRAIN.MODELTYPE)(num_class=config.NumClasses)
    optimizer = optim.SGD(net.parameters(), lr=config.TRAIN.LR, weight_decay=0.001, momentum=0.9)
    if config.TRAIN.MODELPATH is None:
        now_time = datetime.now()
        loss = ''
        for i in config.TRAIN.LOSS:
            loss = loss + '-' + i
        savedir = config.SAVEDIR + config.TRAIN.MODELTYPE + loss + '/' + now_time.strftime('%Y%m%d%H%M%S') + '/'
        log_file = open(savedir + 'log', 'a+')

        trained_epoch = 0
        valid_acc = 0.0
        print('[Message] Create new model.', file=log_file)
        print('[Message] Create folder {}'.format(savedir), file=log_file)
        os.makedirs(savedir)
        os.makedirs(savedir + 'scripts/')
    else:
        savedir = config.TRAIN.MODELPATH.rsplit('/', 1)[0] + '/'
        log_file = open(savedir + 'log', 'a+')

        checkpoint = torch.load(config.TRAIN.MODELPATH)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        trained_epoch = checkpoint['epoch']
        valid_acc = checkpoint['valid_acc']
        print('[Message] Load Model:{}'.format(config.TRAIN.MODELPATH), file=log_file)

    print('[Message] Model type {}'.format(config.TRAIN.MODELTYPE), file=log_file)
    print('[Message] Save dir {}'.format(savedir), file=log_file)

    return net, trained_epoch, optimizer, valid_acc, savedir


def train(net, trained_epoch, optimizer, best_valid_acc, savedir):
    # open the log file
    log_file = open(savedir + 'log', 'a+')

    setup_seed(5)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.VISIBLEDEVICES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print('[Message] Multi GPUS:{}'.format(torch.cuda.device_count()), file=log_file)
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    else:
        print('[Message] 1 GPU', file=log_file)
    net.to(device)

    # Train dataset and dataloader
    train_dataset = SpeechDatasetV2(set='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCHSIZE, shuffle=True,
                                  num_workers=config.TRAIN.NUMWORKS, pin_memory=True)

    # losses
    if 'CE' in config.TRAIN.LOSS:
        # Cross entropy loss
        ce_criterion = nn.CrossEntropyLoss()
    if 'TRIPLET' in config.TRAIN.LOSS:
        # triplet loss
        tri_criterion = nn.TripletMarginLoss()
    if 'CCA' in config.TRAIN.LOSS:
        # CCA loss
        cca_criterion = cca_loss(outdim_size=64, use_all_singular_values=False, device=device).loss

    # optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=config.TRAIN.PATIENCE,
                                                     verbose=True, factor=0.9)

    print('[Train] Batch size {}'.format(config.TRAIN.BATCHSIZE), file=log_file)
    print('[Train] Start epoch {}'.format(trained_epoch), file=log_file)
    print('[Train] Patience {}'.format(config.TRAIN.PATIENCE), file=log_file)
    print('[Train] Init learning Rate {}'.format(config.TRAIN.LR), file=log_file)

    # Counter. If valid acc not improve in patience epochs, stop training
    counter = 0
    best_epoch = 0
    best_state = None

    for epoch in range(1, config.TRAIN.EPOCH + 1):
        net.train()
        with tqdm(train_dataloader, desc='Epoch {}'.format(epoch), ncols=200) as t:
            if config.TRAIN.MODE == 'NoText':
                sum_train_ce_loss = 0
                start_epoch_time = time.time()
                for (waveform, target) in t:
                    start_batch_time = time.time()
                    t.set_description('Epoch {}'.format(epoch))

                    waveform = waveform.type(torch.FloatTensor)
                    waveform, target = waveform.to(device), target.to(device)

                    optimizer.zero_grad()

                    output = net(waveform)

                    train_ce_loss = ce_criterion(output, target)
                    sum_train_ce_loss = sum_train_ce_loss + train_ce_loss.item() * config.TRAIN.BATCHSIZE
                    train_loss = train_ce_loss

                    train_loss.backward()
                    optimizer.step()

                    end_batch_time = time.time()

                    t.set_postfix(train_ce_loss=train_ce_loss.item(),
                                  lr=optimizer.param_groups[0]['lr'],
                                  time=end_batch_time - start_batch_time)
                end_epoch_time = time.time()
                message = '[{}][Train] Epoch:{}, train_ce_loss:{}, lr:{}, train_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    epoch,
                    sum_train_ce_loss / len(train_dataloader.dataset),
                    optimizer.param_groups[0]['lr'],
                    end_epoch_time - start_epoch_time
                )
            elif config.TRAIN.MODE == 'Text':
                sum_train_ce_loss = 0
                sum_train_tri_loss = 0
                start_epoch_time = time.time()
                for (waveform, match_word_vec, unmatch_word_vec, target) in t:
                    start_batch_time = time.time()
                    t.set_description('Epoch {}'.format(epoch))

                    waveform = waveform.type(torch.FloatTensor)
                    match_word_vec = match_word_vec.type(torch.FloatTensor)
                    unmatch_word_vec = unmatch_word_vec.type(torch.FloatTensor)
                    waveform = waveform.to(device)
                    match_word_vec = match_word_vec.to(device)
                    unmatch_word_vec = unmatch_word_vec.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()

                    output, audio_embedding, match_word_vec, unmatch_word_vec = net(waveform, match_word_vec,
                                                                                    unmatch_word_vec)

                    train_ce_loss = ce_criterion(output, target)
                    train_tri_loss = tri_criterion(audio_embedding, match_word_vec, unmatch_word_vec)
                    sum_train_ce_loss = sum_train_ce_loss + train_ce_loss.item() * config.TRAIN.BATCHSIZE
                    sum_train_tri_loss = sum_train_tri_loss + train_tri_loss.item() * config.TRAIN.BATCHSIZE
                    train_loss = 0.5 * train_ce_loss + 0.5 * train_tri_loss

                    train_loss.backward()
                    optimizer.step()

                    end_batch_time = time.time()

                    t.set_postfix(train_ce_loss=train_ce_loss.item(),
                                  train_tri_loss=train_tri_loss.item(),
                                  lr=optimizer.param_groups[0]['lr'],
                                  time=end_batch_time - start_batch_time)
                end_epoch_time = time.time()
                message = '[{}][Train] Epoch:{}, train_ce_loss:{}, train_tri_loss:{}, lr:{}, train_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    epoch,
                    sum_train_ce_loss / len(train_dataloader.dataset),
                    sum_train_tri_loss / len(train_dataloader.dataset),
                    optimizer.param_groups[0]['lr'],
                    end_epoch_time - start_epoch_time
                )
            elif config.TRAIN.MODE == 'TextAnchor':
                sum_train_ce_loss = 0
                sum_train_tri_loss = 0
                start_epoch_time = time.time()
                for (pos_waveform, neg_waveform, pos_word_vec, pos_label) in t:
                    start_batch_time = time.time()
                    t.set_description('Epoch {}'.format(epoch))

                    pos_waveform = pos_waveform.type(torch.FloatTensor)
                    neg_waveform = neg_waveform.type(torch.FloatTensor)
                    pos_word_vec = pos_word_vec.type(torch.FloatTensor)
                    pos_waveform = pos_waveform.to(device)
                    neg_waveform = neg_waveform.to(device)
                    pos_word_vec = pos_word_vec.to(device)
                    pos_label = pos_label.to(device)

                    optimizer.zero_grad()

                    pos, audio_embedding_pos, audio_embedding_neg, text_embedding = net(pos_waveform, neg_waveform,
                                                                                        pos_word_vec)
                    train_ce_loss = ce_criterion(pos, pos_label)
                    train_tri_loss = tri_criterion(text_embedding, audio_embedding_pos, audio_embedding_neg)
                    sum_train_ce_loss = sum_train_ce_loss + train_ce_loss.item() * config.TRAIN.BATCHSIZE
                    sum_train_tri_loss = sum_train_tri_loss + train_tri_loss.item() * config.TRAIN.BATCHSIZE
                    train_loss = 0.5 * train_ce_loss + 0.5 * train_tri_loss

                    train_loss.backward()
                    optimizer.step()

                    end_batch_time = time.time()

                    t.set_postfix(train_ce_loss=train_ce_loss.item(),
                                  train_tri_loss=train_tri_loss.item(),
                                  lr=optimizer.param_groups[0]['lr'],
                                  time=end_batch_time - start_batch_time)
                end_epoch_time = time.time()
                message = '[{}][Train] Epoch:{}, train_ce_loss:{}, train_tri_loss:{}, lr:{}, train_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    epoch,
                    sum_train_ce_loss / len(train_dataloader.dataset),
                    sum_train_tri_loss / len(train_dataloader.dataset),
                    optimizer.param_groups[0]['lr'],
                    end_epoch_time - start_epoch_time
                )
            elif config.TRAIN.MODE == 'CCA':
                sum_train_ce_loss = 0
                sum_train_cca_loss = 0
                start_epoch_time = time.time()
                for (waveform, word_vec, target) in t:
                    start_batch_time = time.time()
                    t.set_description('Epoch {}'.format(epoch))

                    waveform = waveform.type(torch.FloatTensor)
                    word_vec = word_vec.type(torch.FloatTensor)
                    waveform = waveform.to(device)
                    word_vec = word_vec.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()

                    output, audio_embedding, text_embedding = net(waveform, word_vec)

                    train_ce_loss = ce_criterion(output, target)
                    train_cca_loss = cca_criterion(audio_embedding, text_embedding)  # * config.TRAIN.BATCHSIZE
                    sum_train_ce_loss = sum_train_ce_loss + train_ce_loss.item() * config.TRAIN.BATCHSIZE
                    sum_train_cca_loss = sum_train_cca_loss + train_cca_loss
                    train_loss = train_ce_loss + train_cca_loss

                    train_loss.backward()
                    optimizer.step()

                    end_batch_time = time.time()

                    t.set_postfix(train_ce_loss=train_ce_loss.item(),
                                  train_cca_loss=train_cca_loss.item(),
                                  lr=optimizer.param_groups[0]['lr'],
                                  time=end_batch_time - start_batch_time)
                end_epoch_time = time.time()
                message = '[{}][Train] Epoch:{}, train_ce_loss:{}, train_cca_loss:{}, lr:{}, train_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    epoch,
                    sum_train_ce_loss / len(train_dataloader.dataset),
                    sum_train_cca_loss / len(train_dataloader.dataset),
                    optimizer.param_groups[0]['lr'],
                    end_epoch_time - start_epoch_time
                )

        print(message, file=log_file)

        valid_loss, valid_acc = valid(net, device=device, epoch=epoch, log_file=log_file)
        scheduler.step(valid_acc)  # valid acc not increase

        # Update best valid models
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_net = net
            best_valid_model = 'epoch_{}_valid_loss_{:.4f}_acc_{:2f}.pth'.format(best_epoch, best_valid_loss,
                                                                                 best_valid_acc)
            if torch.cuda.device_count() == 1:
                best_state = {'net': best_net.state_dict(), 'optimizer': optimizer.state_dict(),
                              'epoch': best_epoch}
            else:
                best_state = {'net': best_net.module.state_dict(), 'optimizer': optimizer.state_dict(),
                              'epoch': best_epoch}
            counter = 0
        else:
            counter = counter + 1
            print('[Early stopping] {}'.format(counter))
            if counter >= config.TRAIN.EARLYSTOP:
                print('[Message] Early stopping.')
                print('[Message] Best valid acc {:.4f}\n'.format(best_valid_acc))
                break
        print('[Message] Best valid acc {:.4f}\n'.format(best_valid_acc))

    # save model optimizer and trained epoch
    torch.save(best_state, savedir + 'epoch_{}_valid_loss_{:.4f}_acc_{:2f}.pth'.format(
        best_epoch,
        best_valid_loss,
        best_valid_acc))

    # save model code
    shutil.copy(config.ROOTDIR + 'models/' + config.TRAIN.MODELTYPE + '.py', savedir + 'scripts/')
    shutil.copy(config.ROOTDIR + 'utils/' + 'dataloader2classes.py', savedir + 'scripts/')
    shutil.copy(config.ROOTDIR + 'config.py', savedir + 'scripts/')
    shutil.copy(config.ROOTDIR + 'train.py', savedir + 'scripts/')
    shutil.copy(config.ROOTDIR + 'test.py', savedir + 'scripts/')

    test_net(best_net, savedir, device, printnet=False)


def valid(net, device=None, epoch=1, log_file=None):
    net.eval()
    valid_dataset = SpeechDatasetV2('valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.VALID.BATCHSIZE, shuffle=False,
                                  num_workers=4, drop_last=False)

    # losses
    if 'CE' in config.TRAIN.LOSS:
        # Cross entropy loss
        ce_criterion = nn.CrossEntropyLoss()
    if 'TRIPLET' in config.TRAIN.LOSS:
        # triplet loss
        tri_criterion = nn.TripletMarginLoss()
    if 'CCA' in config.TRAIN.LOSS:
        # CCA loss
        cca_criterion = cca_loss(outdim_size=64, use_all_singular_values=False, device=device).loss

    with torch.no_grad():
        net = net.to(device)
        with tqdm(valid_dataloader, desc='Valid', ncols=200) as t:
            if config.TRAIN.MODE == 'NoText':
                sum_valid_ce_loss = 0
                batch_id = 0
                correct = 0
                start_valid_time = time.time()
                for (waveform, target) in t:
                    batch_id += 1

                    waveform = waveform.type(torch.FloatTensor)
                    waveform, target = waveform.to(device), target.to(device)

                    output = net(waveform)

                    valid_ce_loss = ce_criterion(output, target)

                    sum_valid_ce_loss += valid_ce_loss.item() * config.VALID.BATCHSIZE

                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    t.set_postfix(valid_ce_loss=valid_ce_loss.item(), acc=correct / batch_id / config.VALID.BATCHSIZE)
                end_valid_time = time.time()
                valid_acc = correct / len(valid_dataloader.dataset)
                valid_loss = sum_valid_ce_loss / len(valid_dataloader.dataset)
                message = '[{}][Valid] valid_acc:{}, valid_ce_loss:{}, valid_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    valid_acc,
                    sum_valid_ce_loss / len(valid_dataloader.dataset),
                    end_valid_time - start_valid_time
                )
                print(message, file=log_file)
                return valid_loss, valid_acc

            elif config.TRAIN.MODE == 'Text':
                sum_valid_ce_loss = 0
                sum_valid_tri_loss = 0
                batch_id = 0
                correct = 0
                start_valid_time = time.time()
                for (waveform, match_word_vec, unmatch_word_vec, target) in t:
                    batch_id += 1

                    waveform = waveform.type(torch.FloatTensor)
                    match_word_vec = match_word_vec.type(torch.FloatTensor)
                    unmatch_word_vec = unmatch_word_vec.type(torch.FloatTensor)
                    waveform = waveform.to(device)
                    match_word_vec = match_word_vec.to(device)
                    unmatch_word_vec = unmatch_word_vec.to(device)
                    target = target.to(device)

                    output, audio_embedding, match_word_vec, unmatch_word_vec = net(waveform, match_word_vec,
                                                                                    unmatch_word_vec)

                    valid_ce_loss = ce_criterion(output, target)
                    valid_tri_loss = tri_criterion(audio_embedding, match_word_vec, unmatch_word_vec)
                    sum_valid_ce_loss += valid_ce_loss.item() * config.VALID.BATCHSIZE
                    sum_valid_tri_loss += valid_tri_loss.item() * config.VALID.BATCHSIZE

                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    t.set_postfix(valid_ce_loss=valid_ce_loss.item(),
                                  valid_tri_loss=valid_tri_loss.item(),
                                  acc=correct / batch_id / config.VALID.BATCHSIZE)

                end_valid_time = time.time()
                valid_acc = correct / len(valid_dataloader.dataset)
                valid_loss = sum_valid_ce_loss / len(valid_dataloader.dataset) + \
                             sum_valid_tri_loss / len(valid_dataloader.dataset)
                message = '[{}][Valid] valid_acc:{}, valid_ce_loss:{}, valid_tri_loss:{}, valid_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    valid_acc,
                    sum_valid_ce_loss / len(valid_dataloader.dataset),
                    valid_tri_loss / len(valid_dataloader.dataset),
                    end_valid_time - start_valid_time
                )
                print(message, file=log_file)
                return valid_loss, valid_acc

            elif config.TRAIN.MODE == 'TextAnchor':
                sum_valid_ce_loss = 0
                sum_valid_tri_loss = 0
                batch_id = 0
                correct = 0
                start_valid_time = time.time()
                for (pos_waveform, neg_waveform, pos_word_vec, pos_label) in t:
                    batch_id += 1
                    pos_waveform = pos_waveform.type(torch.FloatTensor)
                    neg_waveform = neg_waveform.type(torch.FloatTensor)
                    pos_word_vec = pos_word_vec.type(torch.FloatTensor)
                    pos_waveform = pos_waveform.to(device)
                    neg_waveform = neg_waveform.to(device)
                    pos_word_vec = pos_word_vec.to(device)
                    pos_label = pos_label.to(device)

                    pos, audio_embedding_pos, audio_embedding_neg, text_embedding = net(pos_waveform, neg_waveform,
                                                                                        pos_word_vec)

                    valid_ce_loss = ce_criterion(pos, pos_label)
                    valid_tri_loss = tri_criterion(text_embedding, audio_embedding_pos, audio_embedding_neg)
                    sum_valid_ce_loss = sum_valid_ce_loss + valid_ce_loss.item() * config.TRAIN.BATCHSIZE
                    sum_valid_tri_loss = sum_valid_tri_loss + valid_tri_loss.item() * config.TRAIN.BATCHSIZE

                    pred = pos.max(1, keepdim=True)[1]
                    correct += pred.eq(pos_label.view_as(pred)).sum().item()

                    t.set_postfix(valid_ce_loss=valid_ce_loss.item(),
                                  valid_tri_loss=valid_tri_loss.item(),
                                  acc=correct / batch_id / config.VALID.BATCHSIZE)

                end_valid_time = time.time()
                valid_acc = correct / len(valid_dataloader.dataset)
                valid_loss = sum_valid_ce_loss / len(valid_dataloader.dataset) + \
                             sum_valid_tri_loss / len(valid_dataloader.dataset)
                message = '[{}][Valid] valid_acc:{}, valid_ce_loss:{}, valid_tri_loss:{}, valid_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    valid_acc,
                    sum_valid_ce_loss / len(valid_dataloader.dataset),
                    sum_valid_tri_loss / len(valid_dataloader.dataset),
                    end_valid_time - start_valid_time
                )
                print(message, file=log_file)
                return valid_loss, valid_acc

            elif config.TRAIN.MODE == 'CCA':
                sum_valid_ce_loss = 0
                sum_valid_cca_loss = 0
                batch_id = 0
                correct = 0
                start_valid_time = time.time()
                for (waveform, word_vec, target) in t:
                    batch_id += 1

                    waveform = waveform.type(torch.FloatTensor)
                    word_vec = word_vec.type(torch.FloatTensor)
                    waveform = waveform.to(device)
                    word_vec = word_vec.to(device)
                    target = target.to(device)

                    output, audio_embedding, text_embedding = net(waveform, word_vec)

                    valid_ce_loss = ce_criterion(output, target)
                    valid_cca_loss = cca_criterion(audio_embedding, text_embedding)  # * config.TRAIN.BATCHSIZE
                    sum_valid_ce_loss = sum_valid_ce_loss + valid_ce_loss.item() * config.TRAIN.BATCHSIZE
                    sum_valid_cca_loss = sum_valid_cca_loss + valid_cca_loss

                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    t.set_postfix(valid_ce_loss=valid_ce_loss.item(),
                                  valid_tri_loss=valid_cca_loss,
                                  acc=correct / batch_id / config.VALID.BATCHSIZE)
                end_valid_time = time.time()

                valid_acc = correct / len(valid_dataloader.dataset)
                valid_loss = sum_valid_ce_loss / len(valid_dataloader.dataset) + \
                             sum_valid_cca_loss / len(valid_dataloader.dataset)

                message = '[{}][Valid] valid_acc:{}, valid_ce_loss:{}, valid_cca_loss:{}, valid_time(s):{}'.format(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    valid_acc,
                    sum_valid_ce_loss / len(valid_dataloader.dataset),
                    sum_valid_cca_loss / len(valid_dataloader.dataset),
                    end_valid_time - start_valid_time
                )
                print(message, file=log_file)
                return valid_loss, correct / len(valid_dataloader.dataset)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.VISIBLEDEVICES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, trained_epoch, optimizer, valid_acc, savedir = load_model()
    train(net, trained_epoch, optimizer, valid_acc, savedir)
