from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
# from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from model import My3Model_unsupervise, Encoder3
from cosmo_design import FeatureConstructor, CosmoLoss
import data_pre as data

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
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=101,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200,300',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--choose_model', type=str, default="pretrain_model.pth",
                        help='choose_model')
    parser.add_argument('--num_of_samples', type=int, default=3000,
                        help='num_of_samples')

    # method
    parser.add_argument('--num_positive', type=int, default=9,
                        help='num_positive')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the path according to the environment
    opt.node_path = './save_node_unsupervise/node{}/'.format(opt.node_id)
    opt.model_name = 'lr_{}_decay_{}_bsz_{}_temp_{}_epoch_{}'.\
        format(opt.learning_rate, opt.lr_decay_rate, opt.batch_size, opt.temp, opt.epochs)

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
    train_dataset = data.Multimodal_unlabel_dataset(opt.node_id, opt.num_of_samples)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, 
        pin_memory=True, shuffle=True, drop_last=True)

    return train_loader


def set_model(opt):

    model = My3Model_unsupervise()

    criterion = CosmoLoss(temperature=opt.temp)

    ## load model weights
    ckpt_path = os.path.join("../", opt.choose_model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("encoder.", "")
            # print(k)
            if "classifier" not in k:## only append weight of encoders
                new_state_dict[k] = v
        state_dict = new_state_dict
    model.encoder.load_state_dict(state_dict)


    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (input_data1, input_data2, input_data3) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            input_data3 = input_data3.cuda()
        bsz = input_data1.shape[0]

        # compute loss
        feature1, feature2, feature3 = model(input_data1, input_data2, input_data3)

        features = FeatureConstructor(feature1, feature2, feature3, opt.num_positive)

        loss = criterion(features)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


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
    server_addr = "172.22.172.75"#"10.54.20.19"
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
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)
    w_parameter_init = get_model_array(model)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    record_loss = np.zeros(opt.epochs)

    compute_time_record = np.zeros(opt.epochs)
    upper_commu_time_record = np.zeros(int(opt.epochs/opt.fl_epoch))
    down_commu_time_record = np.zeros(int(opt.epochs/opt.fl_epoch))

    all_time_record = np.zeros(opt.epochs + 2)
    all_time_record[0] = time.time()

    begin_time = time.time()

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        record_loss[epoch-1] = loss
        compute_time_record[epoch-1] = time2 - time1

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
            reset_model_parameter(new_w, model)
            w_parameter_init = new_w

        # save model
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        all_time_record[epoch] = time.time()

        np.savetxt(opt.result_path + "record_loss.txt", record_loss)
        np.savetxt(opt.result_path + "compute_time_record.txt", compute_time_record)
        np.savetxt(opt.result_path + "upper_commu_time_record.txt", upper_commu_time_record)
        np.savetxt(opt.result_path + "down_commu_time_record.txt", down_commu_time_record)
        np.savetxt(opt.result_path + "all_time_record.txt", all_time_record)

    end_time = time.time()
    all_time_record[epoch+1] = end_time - begin_time
    print("Total training delay: ", end_time - begin_time)

    comm.disconnect(1)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    np.savetxt(opt.result_path + "record_loss.txt", record_loss)
    np.savetxt(opt.result_path + "compute_time_record.txt", compute_time_record)
    np.savetxt(opt.result_path + "upper_commu_time_record.txt", upper_commu_time_record)
    np.savetxt(opt.result_path + "down_commu_time_record.txt", down_commu_time_record)
    np.savetxt(opt.result_path + "all_time_record.txt", all_time_record)
    
    

if __name__ == '__main__':
    main()
