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
from utils.dataloader import GoogleSpeechCommandDataset
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import config
import models
import random
import sys
from tqdm import tqdm
import torch.nn as nn
import logging
from utils.cca_loss import cca_loss
import time
from sklearn.metrics import auc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model():
    net = getattr(models, config.TEST.MODELTYPE)()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if os.path.exists(config.TEST.MODELPATH):
        savedir = config.TEST.MODELPATH.rsplit('/', 1)[0] + '/'

        handler = logging.FileHandler(savedir + 'log.log')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)

        checkpoint = torch.load(config.TEST.MODELPATH)
        net.load_state_dict(checkpoint['net'])
        logger.info('[TEST] Load Model:{}'.format(config.TEST.MODELPATH))
    else:
        logger.info('[ERROR] Model does not exist, please check config.TEST.MODELPATH')
        exit(1)
    return net, savedir, logger


def cal_nums(output, target, thres):
    '''
    calculate TP FP FN TN in a batch
    calculate for each class, which means regard one class as positive (1), other as negative (0)
    :param output: network output. torch.Size([batch_size, num_class])
    :param target: ground truth. torch.Size([batch_size])
    :param thres: thresholds list [0.1 0.2 0.3 ....]
    :return: TP FP FN TN in the batch
    '''

    # (num_class, 4matrix) 4matrix : TP + FP + FN + TN
    result = np.zeros((len(thres), output.size()[1], 4))

    # convert target to torch.Size([batch_size,1])
    target = torch.unsqueeze(target, dim=1)
    # print(target)

    # one hot vector target torch.Size([batch_size, num_class])
    target_onehot = torch.zeros(output.size()).scatter_(1, target, 1)

    # output softmax
    output = F.softmax(output, dim=1)

    # convert tensor to array
    output = output.numpy()
    target_onehot = target_onehot.numpy()

    for i, thre in enumerate(thres):
        for n in range(output.shape[1]):
            # for one class
            TP = FP = FN = TN = 0
            for m in range(output.shape[0]):
                proba = output[m, n]
                if proba >= thre:
                    proba = 1
                if proba == 1:
                    if target_onehot[m, n] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if target_onehot[m, n] == 0:
                        TN += 1
                    else:
                        FN += 1
            if TP + FP + FN + TN != output.shape[0]:
                print('[Error] Calculate TP FP FN TN wrong!', flush=True)
                exit(1)

            result[i, n, 0] = TP
            result[i, n, 1] = FP
            result[i, n, 2] = FN
            result[i, n, 3] = TN

    return result


def cal_rates(result):
    '''
    calculate
    :param result: (thres, num_class, TP FP FN TN)
    :return:
    '''

    TP = result[:, :, 0]  # (thres, num_class)
    FP = result[:, :, 1]
    FN = result[:, :, 2]
    TN = result[:, :, 3]

    TPR = np.true_divide(TP, TP + FN)
    TNR = np.true_divide(TN, FP + TN)
    FNR = np.true_divide(FN, TP + FN)
    FPR = np.true_divide(FP, FP + TN)

    return TPR, TNR, FNR, FPR


def find_close(arr, e):
    '''
    Find the closest val in arr of e
    :param arr: numpy array
    :param e:value e
    :return:closest value arr[idx] and index idx
    '''
    size = len(arr)
    idx = 0
    val = abs(e - arr[idx])

    for i in range(1, size):
        val1 = abs(e - arr[i])
        if val1 < val:
            idx = i
            val = val1

    return arr[idx], idx


def test_net(net, savedir, logger, mode):
    net.eval()
    setup_seed(config.SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TEST.VISIBLEDEVICES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = GoogleSpeechCommandDataset(set='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.TEST.BATCHSIZE, shuffle=False, num_workers=4,
                                 drop_last=False)

    class_correct = list(0. for i in range(config.NumClasses))
    class_total = list(0. for i in range(config.NumClasses))

    acc_file = open(savedir + '/test_acc.txt', 'w+')
    acc_file.truncate()

    losses = savedir.rsplit('/', 2)[0].split('-')
    if 'CE' in losses:
        # Cross entropy loss
        ce_criterion = nn.CrossEntropyLoss()
    if 'TRIPLET' in losses:
        # triplet loss
        tri_criterion = nn.TripletMarginLoss()
    if 'CCA' in losses:
        # CCA loss
        cca_criterion = cca_loss(outdim_size=config.TRAIN.CCAOUTDIM, use_all_singular_values=False, device=device).loss

    with torch.no_grad():
        net = net.to(device)
        classes = test_dataset.commands
        classes.append('_unknown_')
        # thresholds = np.concatenate(
        #     (np.linspace(0.0, 0.2, 1000), np.linspace(0.2, 0.8, 500), np.linspace(0.8, 1.0, 1000)))
        thresholds = np.linspace(0.0, 1.0, 100)
        thresholds = thresholds.tolist()
        result = np.zeros((len(thresholds), len(classes), 4))  # TP + FP + FN + TN

        with tqdm(test_dataloader, desc='Test', ncols=150) as t:
            if mode == 'NoText':
                sum_test_ce_loss = 0
                batch_id = 0
                correct = 0
                start_test_time = time.time()
                for (waveform, target) in t:
                    batch_id += 1

                    waveform = waveform.type(torch.FloatTensor)
                    waveform, target = waveform.to(device), target.to(device)

                    output = net(waveform)

                    test_ce_loss = ce_criterion(output, target)

                    sum_test_ce_loss += test_ce_loss.item() * config.TEST.BATCHSIZE

                    pred = output.max(1, keepdim=True)[1]

                    _, predicted = torch.max(output, 1)
                    c = (predicted == target).squeeze()
                    for i in range(config.TEST.BATCHSIZE):
                        tar = target[i]
                        class_correct[tar] += c[i].item()
                        class_total[tar] += 1
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    t.set_postfix(acc=correct / batch_id / config.TEST.BATCHSIZE, test_ce_loss=test_ce_loss.item())

                    result = result + cal_nums(output.cpu(), target.cpu(), thresholds)
                end_test_time = time.time()

                test_acc = correct / len(test_dataloader.dataset)
                message = '[Test] test_acc:{:4f}, test_ce_loss:{:4f}, test_time(s):{:4f}'.format(
                    test_acc,
                    sum_test_ce_loss / len(test_dataloader.dataset),
                    end_test_time - start_test_time
                )
                logger.info(message)

            elif mode == 'Text':
                sum_test_ce_loss = 0
                sum_test_tri_loss = 0
                batch_id = 0
                correct = 0
                start_test_time = time.time()
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

                    test_ce_loss = ce_criterion(output, target)
                    test_tri_loss = tri_criterion(audio_embedding, match_word_vec, unmatch_word_vec)
                    sum_test_ce_loss += test_ce_loss.item() * config.TEST.BATCHSIZE
                    sum_test_tri_loss += test_tri_loss.item() * config.TEST.BATCHSIZE

                    pred = output.max(1, keepdim=True)[1]
                    _, predicted = torch.max(output, 1)
                    c = (predicted == target).squeeze()
                    for i in range(config.TEST.BATCHSIZE):
                        tar = target[i]
                        class_correct[tar] += c[i].item()
                        class_total[tar] += 1

                    correct += pred.eq(target.view_as(pred)).sum().item()
                    t.set_postfix(acc=correct / batch_id / config.TEST.BATCHSIZE, test_ce_loss=test_ce_loss.item(),
                                  test_tri_loss=test_tri_loss.item())

                    result = result + cal_nums(output.cpu(), target.cpu(), thresholds)
                end_test_time = time.time()
                test_acc = correct / len(test_dataloader.dataset)

                message = '[Test] test_acc:{:4f}, test_ce_loss:{:4f}, test_tri_loss:{:4f}, test_time(s):{:4f}'.format(
                    test_acc,
                    sum_test_ce_loss / len(test_dataloader.dataset),
                    test_tri_loss / len(test_dataloader.dataset),
                    end_test_time - start_test_time
                )
                logger.info(message)

            elif mode == 'TextAnchor':
                sum_test_ce_loss = 0
                sum_test_tri_loss = 0
                batch_id = 0
                correct = 0
                start_test_time = time.time()
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

                    test_ce_loss = ce_criterion(pos, pos_label)
                    test_tri_loss = tri_criterion(text_embedding, audio_embedding_pos, audio_embedding_neg)
                    sum_test_ce_loss = sum_test_ce_loss + test_ce_loss.item() * config.TEST.BATCHSIZE
                    sum_test_tri_loss = sum_test_tri_loss + test_tri_loss.item() * config.TEST.BATCHSIZE

                    pred = pos.max(1, keepdim=True)[1]
                    _, predicted = torch.max(pos, 1)
                    c = (predicted == pos_label).squeeze()
                    for i in range(config.TEST.BATCHSIZE):
                        tar = pos_label[i]
                        class_correct[tar] += c[i].item()
                        class_total[tar] += 1

                    correct += pred.eq(pos_label.view_as(pred)).sum().item()
                    t.set_postfix(acc=correct / batch_id / config.TEST.BATCHSIZE,
                                  test_ce_loss=test_ce_loss.item(),
                                  test_tri_loss=test_tri_loss.item())

                    result = result + cal_nums(pos.cpu(), pos_label.cpu(), thresholds)
                end_test_time = time.time()
                test_acc = correct / len(test_dataloader.dataset)

                message = '[Test] test_acc:{:4f}, test_ce_loss:{:4f}, test_tri_loss:{:4f}, test_time(s):{:4f}'.format(
                    test_acc,
                    sum_test_ce_loss / len(test_dataloader.dataset),
                    test_tri_loss / len(test_dataloader.dataset),
                    end_test_time - start_test_time
                )
                logger.info(message)

            elif mode == 'CCA':
                sum_test_ce_loss = 0
                sum_test_cca_loss = 0
                batch_id = 0
                correct = 0
                start_test_time = time.time()
                for (waveform, word_vec, target) in t:
                    batch_id += 1

                    waveform = waveform.type(torch.FloatTensor)
                    word_vec = word_vec.type(torch.FloatTensor)
                    waveform = waveform.to(device)
                    word_vec = word_vec.to(device)
                    target = target.to(device)

                    output, audio_embedding, text_embedding = net(waveform, word_vec)

                    test_ce_loss = ce_criterion(output, target)
                    test_cca_loss = cca_criterion(audio_embedding, text_embedding)  # * config.TEST.BATCHSIZE
                    sum_test_ce_loss = sum_test_ce_loss + test_ce_loss.item() * config.TEST.BATCHSIZE
                    sum_test_cca_loss = sum_test_cca_loss + test_cca_loss

                    pred = output.max(1, keepdim=True)[1]
                    _, predicted = torch.max(output, 1)
                    c = (predicted == target).squeeze()
                    for i in range(config.TEST.BATCHSIZE):
                        tar = target[i]
                        class_correct[tar] += c[i].item()
                        class_total[tar] += 1

                    correct += pred.eq(target.view_as(pred)).sum().item()
                    t.set_postfix(acc=correct / batch_id / config.TEST.BATCHSIZE,
                                  test_ce_loss=test_ce_loss.item(),
                                  test_cca_loss=test_cca_loss)

                    result = result + cal_nums(output.cpu(), target.cpu(), thresholds)
                end_test_time = time.time()
                test_acc = correct / len(test_dataloader.dataset)

                message = '[Test] test_acc:{:4f}, test_ce_loss:{:4f}, test_tri_loss:{:4f}, test_time(s):{:4f}'.format(
                    test_acc,
                    sum_test_ce_loss / len(test_dataloader.dataset),
                    sum_test_cca_loss / len(test_dataloader.dataset),
                    end_test_time - start_test_time
                )
                logger.info(message)

            elif mode == 'ThreeAudios':
                net.eval()
                sum_test_ce_loss = 0
                sum_test_tri_loss = 0
                batch_id = 0
                correct = 0
                start_test_time = time.time()
                for (anchor_waveform, pos_waveform, neg_waveform, anchor_label_num) in t:
                    batch_id += 1
                    anchor_waveform = anchor_waveform.type(torch.FloatTensor)
                    pos_waveform = pos_waveform.type(torch.FloatTensor)
                    neg_waveform = neg_waveform.type(torch.FloatTensor)
                    anchor_waveform = anchor_waveform.to(device)
                    pos_waveform = pos_waveform.to(device)
                    neg_waveform = neg_waveform.to(device)
                    anchor_label_num = anchor_label_num.to(device)

                    anchor_output, audio_embedding_anchor, audio_embedding_pos, audio_embedding_neg = net(
                        anchor_waveform, pos_waveform, neg_waveform)

                    test_ce_loss = ce_criterion(anchor_output, anchor_label_num)
                    test_tri_loss = tri_criterion(audio_embedding_anchor, audio_embedding_pos, audio_embedding_neg)
                    sum_test_ce_loss = sum_test_ce_loss + test_ce_loss.item() * config.TEST.BATCHSIZE
                    sum_test_tri_loss = sum_test_tri_loss + test_tri_loss.item() * config.TEST.BATCHSIZE

                    pred = anchor_output.max(1, keepdim=True)[1]
                    _, predicted = torch.max(anchor_output, 1)
                    c = (predicted == anchor_label_num).squeeze()
                    for i in range(config.TEST.BATCHSIZE):
                        tar = anchor_label_num[i]
                        class_correct[tar] += c[i].item()
                        class_total[tar] += 1

                    correct += pred.eq(anchor_label_num.view_as(pred)).sum().item()
                    t.set_postfix(acc=correct / batch_id / config.TEST.BATCHSIZE,
                                  test_ce_loss=test_ce_loss.item(),
                                  test_tri_loss=test_tri_loss.item())

                    result = result + cal_nums(anchor_output.cpu(), anchor_label_num.cpu(), thresholds)
                end_test_time = time.time()
                test_acc = correct / len(test_dataloader.dataset)

                message = '[Test] test_acc:{:4f}, test_ce_loss:{:4f}, test_tri_loss:{:4f}, test_time(s):{:4f}'.format(
                    test_acc,
                    sum_test_ce_loss / len(test_dataloader.dataset),
                    test_tri_loss / len(test_dataloader.dataset),
                    end_test_time - start_test_time
                )
                logger.info(message)


        logger.info('[Test] Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_dataloader.dataset),
                                                                100. * correct / len(test_dataloader.dataset)))
        acc_file.write('[Test] Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_dataloader.dataset),
                                                                   100. * correct / len(test_dataloader.dataset)))
        unknown_index = classes.index('_unknown_')
        logger.info('[Test] Accuracy without unknown: {}/{} ({:.2f}%)\n'
                    .format(correct - class_correct[unknown_index],
                            len(test_dataloader.dataset) - class_total[unknown_index],
                            100. * (correct - class_correct[unknown_index])
                            / (len(test_dataloader.dataset) - class_total[unknown_index])))
        acc_file.write('[Test] Accuracy without unknow: {}/{} ({:.2f}%)\n'
                       .format(correct - class_correct[unknown_index],
                               len(test_dataloader.dataset) - class_total[unknown_index],
                               100. * (correct - class_correct[unknown_index])
                               / (len(test_dataloader.dataset) - class_total[unknown_index])))

        for i in range(config.NumClasses):
            if class_total[i] == 0:
                logger.info('[Test] Accuracy of {} : {:.2f}% ({}/{})'.format(classes[i], 100 * class_correct[i],
                                                                             int(class_correct[i]),
                                                                             int(class_total[i])))
                acc_file.write('[Test] Accuracy of {} : {:.2f}% ({}/{})\n'.format(classes[i], 100 * class_correct[i],
                                                                                  int(class_correct[i]),
                                                                                  int(class_total[i])))
            else:
                logger.info(
                    '[Test] Accuracy of {} : {:.2f}% ({}/{})'.format(classes[i],
                                                                     100 * class_correct[i] / class_total[i],
                                                                     int(class_correct[i]), int(class_total[i])))
                acc_file.write(
                    '[Test] Accuracy of {} : {:.2f}% ({}/{})\n'.format(classes[i],
                                                                       100 * class_correct[i] / class_total[i],
                                                                       int(class_correct[i]), int(class_total[i])))

        TPR, TNR, FNR, FPR = cal_rates(result)
        TPR = np.mean(TPR, 1)
        TNR = np.mean(TNR, 1)
        FNR = np.mean(FNR, 1)
        FPR = np.mean(FPR, 1)

        AUC = auc(FPR, TPR)
        logger.info('[Test] AUC:{}'.format(AUC))
        acc_file.write('[Test] AUC:{}'.format(AUC))

        test_result = {}
        test_result['FPR'] = FPR
        test_result['FNR'] = FNR
        test_result['TPR'] = TPR
        test_result['TNR'] = TNR
        test_result['AUC'] = AUC

        if os.path.exists(savedir + '/test_result.npy'):
            # Delete if test_result.npy already exists
            os.remove(savedir + '/test_result.npy')
        np.save(savedir + '/test_result.npy', test_result)

        acc_file.close()

        plt.plot(FPR, FNR, label=config.TEST.MODELTYPE)
        plt.legend()
        plt.xlabel('False Alarm Rate')
        plt.ylabel('False Reject Rate')
        plt.xlim((0, 0.1))
        plt.ylim((0, 0.1))
        plt.title('DET')
        plt.show()
        plt.savefig(savedir + '/DET.png')
        logger.info('[Test] Success save DET.png at :{}'.format(savedir))


if __name__ == '__main__':
    net, savedir, logger = load_model()
    test_net(net, savedir, logger, config.TEST.MODE)
