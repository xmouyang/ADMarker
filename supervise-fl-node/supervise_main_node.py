from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

# import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
# from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model

from model import MySingleModel, My3Model
import data_pre as data
from sklearn.metrics import f1_score

from communication import COMM


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--node_id', type=int, default=0,
                        help='node_id')
    parser.add_argument('--fl_epoch', type=int, default=10,
                    help='communication to server after the epoch of local training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--test_print_freq', type=int, default=5,
                        help='test print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=101,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200,300',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--kl_lamda', type=float, default=0.5,
                        help='kl_lamda')

    # model dataset
    parser.add_argument('--choose_model', type=str, default="last.pth",
                        help='choose_model')
    parser.add_argument('--load_model', type=str, default="encoder",
                        help='load_model')
    parser.add_argument('--num_class', type=int, default=16,
                        help='num_class')
    parser.add_argument('--num_of_samples', type=int, default=1000,
                        help='num_of_samples')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()


    opt.num_per_class, opt.train_labels, opt.num_of_train = data.count_num_per_class(opt.node_id, opt.num_class, opt.num_of_samples)
    print("num of train data:", len(opt.train_labels))
    print("train data num_per_class:", opt.num_per_class)


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the path according to the environment
    opt.model_path = '../unsupervise-fl-node/save_node_unsupervise/node{}'.format(opt.node_id)
    opt.load_model_name = 'lr_0.01_decay_0.9_bsz_16_temp_0.07_epoch_101/models/'
    if opt.cosine:
        opt.load_model_name = '{}_cosine'.format(opt.load_model_name)
    opt.load_folder = os.path.join(opt.model_path, opt.load_model_name)

    opt.load_folder_classifier = '../pretrain_model.pth'

    opt.node_path = './save_node_supervise/node{}/'.format(opt.node_id)
    opt.model_name = 'lr_{}_decay_{}_bsz_{}_epoch_{}'.\
        format(opt.learning_rate, opt.lr_decay_rate, opt.batch_size, opt.epochs)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.node_path, opt.model_name, 'models')
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.result_path = os.path.join(opt.node_path, opt.model_name, 'results/')
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt


def set_loader(opt):
   
    # construct data loader

    # load data (already normalized)
    train_dataset = data.Multimodal_train_dataset(opt.node_id, opt.num_of_samples)
    test_dataset = data.Multimodal_test_dataset(opt.node_id)

    #re-sampling
    labels = opt.train_labels
    sample_weight = [1/opt.num_per_class[labels[i]] for i in range(len(labels))]
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weight, num_samples=opt.num_of_train, replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, sampler=sampler,
        pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, 
        pin_memory=True, shuffle=True, drop_last=True)

    return train_loader, test_loader


def set_model(opt):

    model = My3Model(num_classes=opt.num_class)

    ## define loss functions
    criterion = torch.nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    ## load model weights
    ckpt_path = os.path.join(opt.load_folder, opt.choose_model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("encoder.", "")
            # print(k)
            if "head" not in k:## only append weight of encoders
                new_state_dict[k] = v
        state_dict = new_state_dict
    model.encoder.load_state_dict(state_dict)


    ## load model weights of classifier
    if opt.load_model == "all":
        ckpt_classifier = torch.load(opt.load_folder_classifier, map_location='cpu')
        state_dict_classifier = ckpt_classifier['model']

        if torch.cuda.is_available():
            new_state_dict = {}
            for k, v in state_dict_classifier.items():
                k = k.replace("module.", "")
                # print(k)
                if "classifier" in k:## only append weight of classifier
                    k = k.replace("classifier.", "")
                    new_state_dict[k] = v
            state_dict_classifier = new_state_dict
        model.classifier.load_state_dict(state_dict_classifier)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, kl_criterion


def set_global_model(opt):

    model = My3Model(num_classes = opt.num_class)

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    return model

def train_multi(train_loader, model, global_model, criterion, kl_criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1_meter = AverageMeter()
    f1score_meter = AverageMeter()
    confusion = np.zeros((opt.num_class, opt.num_class))

    end = time.time()

    for batch_idx, (input_data1, input_data2, input_data3, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            input_data3 = input_data3.cuda()
            labels = labels.cuda()
        bsz = input_data1.shape[0]

        # compute loss
        output = model(input_data1, input_data2, input_data3)


        if epoch > opt.fl_epoch:
            output_global = global_model(input_data1, input_data2, input_data3)
            print(criterion(output, labels))
            print(opt.kl_lamda * kl_criterion(output, output_global.detach()))
            loss = criterion(output, labels) + opt.kl_lamda * kl_criterion(output, output_global.detach())
        else:
            loss = criterion(output, labels)


        losses.update(loss.item(), bsz)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        batch_f1 = f1_score(labels.cpu().numpy(), output.max(1)[1].cpu().numpy(), average="weighted")

        # calculate and store confusion matrix
        rows = labels.cpu().numpy()
        cols = output.max(1)[1].cpu().numpy()
        for sample_index in range(labels.shape[0]):
            confusion[rows[sample_index], cols[sample_index]] += 1
        top1_meter.update(acc5[0], bsz)
        f1score_meter.update(batch_f1, bsz)


        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print(f1score.val, f1score.avg)

        # print info
        if (batch_idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                  'F1 {f1score.val:.3f} ({f1score.avg:.3f})\t'.format(
                   epoch, batch_idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1 = top1_meter, f1score = f1score_meter))
            sys.stdout.flush()

    top1 = top1_meter.avg
    f1score = f1score_meter.avg

    return losses.avg, top1, f1score, confusion



def validate_multi(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1_meter = AverageMeter()
    f1score_meter = AverageMeter()
    confusion = np.zeros((opt.num_class, opt.num_class))


    with torch.no_grad():
        end = time.time()
        for batch_idx, (input_data1, input_data2, input_data3, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda()
                input_data2 = input_data2.float().cuda()
                input_data3 = input_data3.float().cuda()
                labels = labels.cuda()
            bsz = input_data1.shape[0]

            # forward
            output = model(input_data1, input_data2, input_data3)
            loss = criterion(output, labels)
            losses.update(loss.item(), bsz)

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            batch_f1 = f1_score(labels.cpu().numpy(), output.max(1)[1].cpu().numpy(), average="weighted")

            # calculate and store confusion matrix
            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()
            for sample_index in range(labels.shape[0]):
                confusion[rows[sample_index], cols[sample_index]] += 1
            top1_meter.update(acc5[0], bsz)
            f1score_meter.update(batch_f1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % opt.test_print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                      'F1 {f1score.val:.3f} ({f1score.avg:.3f})\t'.format(
                       batch_idx, len(val_loader), batch_time=batch_time, loss=losses, top1 = top1_meter, f1score = f1score_meter))

    top1 = top1_meter.avg
    f1score = f1score_meter.avg

    return losses.avg, top1, f1score, confusion


def get_model_array(model):

    params = []
    for param in model.parameters():
        if torch.cuda.is_available():
            params.extend(param.view(-1).cpu().detach().numpy())
        else:
            params.extend(param.view(-1).detach().numpy())
        # print(param)

    # model_params = params.cpu().numpy()
    model_params = np.array(params)
    print("Shape of model weight: ", model_params.shape)#39456

    return model_params



def reset_model_parameter(new_params, model):

    temp_index = 0

    with torch.no_grad():
        for param in model.parameters():

            # print(param.shape)

            if len(param.shape) == 2:

                para_len = int(param.shape[0] * param.shape[1])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1])))
                temp_index += para_len

            elif len(param.shape) == 3:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2])))
                temp_index += para_len 

            elif len(param.shape) == 4:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])))
                temp_index += para_len  

            elif len(param.shape) == 5:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3] * param.shape[4])
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])))
                temp_index += para_len  

            else:

                para_len = param.shape[0]
                temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight))
                temp_index += para_len                   


def set_commu(opt):

    #prepare the communication module
    server_addr = "172.22.172.75"
    # server_addr = "localhost"

    server_port = 30415

    comm = COMM(server_addr,server_port, opt.node_id)

    comm.send2server('hello',-1)

    print(comm.recvfserver())

    return comm


def main():

    opt = parse_option()

    # set up communication with sevrer
    comm = set_commu(opt)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion, kl_criterion = set_model(opt)
    w_parameter_init = get_model_array(model)

    # build global model
    global_model = set_global_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)
    record_f1 = np.zeros(opt.epochs)

    compute_time_record = np.zeros(opt.epochs)
    upper_commu_time_record = np.zeros(int(opt.epochs/opt.fl_epoch))
    down_commu_time_record = np.zeros(int(opt.epochs/opt.fl_epoch))

    all_time_record = np.zeros(opt.epochs + 2)
    all_time_record[0] = time.time()

    best_acc = 0
    best_f1 = 0
    best_confusion = np.zeros((opt.num_class, opt.num_class))

    begin_time = time.time()

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc, train_f1_score, train_confusion = train_multi(train_loader, model, global_model, criterion, kl_criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        record_loss[epoch-1] = loss
        compute_time_record[epoch-1] = time2 - time1

        ## record acc
        # if opt.local_modality == "all":
        loss, val_acc, val_f1_score, val_confusion = validate_multi(val_loader, model, criterion, opt)
        
        record_acc[epoch-1] = val_acc
        record_f1[epoch-1] = val_f1_score

        if best_acc < val_acc:
            best_acc = val_acc
        if best_f1 < val_f1_score:
            best_f1 = val_f1_score

        # communication with the server every fl_epoch 
        if (epoch % opt.fl_epoch) == 0:

            ## send model update to the server
            print("Node {} sends weight to the server:".format(opt.node_id))
            w_parameter = get_model_array(model) #obtain the model parameters or gradients 
            w_update = w_parameter - w_parameter_init

            comm_time1 = time.time()
            comm.send2server(w_update,0)
            comm_time2 = time.time()
            commu_epoch = int(epoch/opt.fl_epoch - 1)
            upper_commu_time_record[commu_epoch] = comm_time2 - comm_time1
            print("time for sending model weights:", comm_time2 - comm_time1)

            ## recieve aggregated model update from the server
            comm_time3 = time.time()
            new_w_update, sig_stop = comm.recvOUF()
            comm_time4 = time.time()
            down_commu_time_record[commu_epoch] = comm_time4 - comm_time3
            print("time for downloading model weights:", comm_time4 - comm_time3)
            print("Received weight from the server:", new_w_update.shape)
            # print("Received signal from the server:", sig_stop)
            
            ## update the model according to the received weights
            new_w = w_parameter_init + new_w_update
            # reset_model_parameter(new_w, model)
            reset_model_parameter(new_w, global_model)#do not replace the local model, only use global model to guide its training
            w_parameter_init = new_w

        # save model
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        all_time_record[epoch] = time.time()

        np.savetxt(opt.result_path + "record_loss.txt", record_loss)
        np.savetxt(opt.result_path + "record_acc.txt", record_acc)
        np.savetxt(opt.result_path + "record_f1.txt", record_f1)
        np.savetxt(opt.result_path + "compute_time_record.txt", compute_time_record)
        np.savetxt(opt.result_path + "upper_commu_time_record.txt", upper_commu_time_record)
        np.savetxt(opt.result_path + "down_commu_time_record.txt", down_commu_time_record)
        np.savetxt(opt.result_path + "all_time_record.txt", all_time_record)

    end_time = time.time()
    all_time_record[epoch+1] = end_time - begin_time
    print("Total training delay: ", end_time - begin_time)

    print("best_acc:", best_acc)
    print("best_f1:", best_f1)


    comm.disconnect(1)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    np.savetxt(opt.result_path + "record_loss.txt", record_loss)
    np.savetxt(opt.result_path + "record_acc.txt", record_acc)
    np.savetxt(opt.result_path + "record_f1.txt", record_f1)
    np.savetxt(opt.result_path + "compute_time_record.txt", compute_time_record)
    np.savetxt(opt.result_path + "upper_commu_time_record.txt", upper_commu_time_record)
    np.savetxt(opt.result_path + "down_commu_time_record.txt", down_commu_time_record)
    np.savetxt(opt.result_path + "all_time_record.txt", all_time_record)
    
    

if __name__ == '__main__':
    main()
