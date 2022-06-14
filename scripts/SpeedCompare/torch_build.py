# -*- coding: utf-8 -*-
'''
Build models and trainging scripyt
We provide three different functions for training different models.
Where train_CNN is used to train CNN, train_vCNN is used to train vCNN, and train_vCNNSEL is used to specifically train vCNN and save the results of each step to show the change of Shannon entropy.
train_CNN(...)
train_vCNN(...)
train_vCNNSEL(...)
'''
# conda install h5py tensorboard scikit-learn matplotlib
import pdb
import sys
import math
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import random
import pickle
import glob

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
# Our module!
sys.path.append("../../core")
from MarkonvCore import *



def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)

def buildModel(kernel_size, number_of_kernel, outputdim,modeltype):
    """

    Args:
        kernel_size:
        number_of_kernel:
        outputdim:
        modeltype:

    Returns:

    """
    if modeltype=="Markonv":
        net = MarkonvVModel(int(kernel_size*1.5), number_of_kernel, outputdim)
    return net

class MarkonvVModel(nn.Module):
    def __init__(self, kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.markonv = MarkonvV(kernel_size, number_of_kernel, 4)
        self.dropout = nn.Dropout(p=0.2)
        self.outputdim = outputdim
        self.linear = torch.nn.Linear(number_of_kernel, outputdim)
    def forward(self,x):
        mar = self.markonv(x)
        pooling1 = torch.max(mar,dim=2).values #keras.layers.GlobalMaxPooling1D()(x)
        output = self.dropout(pooling1) #keras.layers.core.Dropout(0.2)(pooling1)
        if self.outputdim:
            output = self.linear(output) #tf.keras.layers.Dense(outputdim)(output)
            output = torch.sigmoid(output) #keras.layers.Activation("sigmoid")(output)
        return output.squeeze()

class MarkonvRModel(nn.Module):
    def __init__(self, kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.markonv = MarkonvR(kernel_size, number_of_kernel, 4)
        self.dropout = nn.Dropout(p=0.2)
        self.outputdim = outputdim
        self.linear = torch.nn.Linear(number_of_kernel, outputdim)
    def forward(self,x):
        mar = self.markonv(x)
        pooling1 = torch.max(mar,dim=2).values #keras.layers.GlobalMaxPooling1D()(x)
        output = self.dropout(pooling1) #keras.layers.core.Dropout(0.2)(pooling1)
        if self.outputdim:
            output = self.linear(output) #tf.keras.layers.Dense(outputdim)(output)
            output = torch.sigmoid(output) #keras.layers.Activation("sigmoid")(output)
        return output.squeeze()


class MarkonvModel(nn.Module):
    def __init__(self, kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.markonv = Markonv(kernel_size, number_of_kernel, 4)
        self.dropout = nn.Dropout(p=0.2)
        self.outputdim = outputdim
        self.linear = torch.nn.Linear(number_of_kernel, outputdim)
    def forward(self,x):
        mar = self.markonv(x)
        pooling1 = torch.max(mar,dim=2).values #keras.layers.GlobalMaxPooling1D()(x)
        output = self.dropout(pooling1) #keras.layers.core.Dropout(0.2)(pooling1)
        if self.outputdim:
            output = self.linear(output) #tf.keras.layers.Dense(outputdim)(output)
            output = torch.sigmoid(output) #keras.layers.Activation("sigmoid")(output)
        return output.squeeze()



class CNN(nn.Module):
    def __init__(self, in_channels,kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.Conv1d = torch.nn.Conv1d(in_channels, number_of_kernel, kernel_size)
        self.dropout = nn.Dropout(p=0.2)
        self.outputdim = outputdim
        self.linear = torch.nn.Linear(number_of_kernel, outputdim)
    def forward(self,x):
        mar = self.Conv1d(x)
        pooling1 = torch.max(mar,dim=2).values #keras.layers.GlobalMaxPooling1D()(x)
        output = self.dropout(pooling1) #keras.layers.core.Dropout(0.2)(pooling1)
        if self.outputdim:
            output = self.linear(output) #tf.keras.layers.Dense(outputdim)(output)
            output = torch.sigmoid(output) #keras.layers.Activation("sigmoid")(output)
        return output.squeeze()

class HOCNNLB(nn.Module):
    def __init__(self, in_channels,kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.Conv1d = torch.nn.Conv1d(in_channels, number_of_kernel, kernel_size)
        self.dropout = nn.Dropout(p=0.2)
        self.outputdim = outputdim
        self.linear = torch.nn.Linear(number_of_kernel, outputdim)
    def forward(self,x):
        mar = self.Conv1d(x)
        pooling1 = torch.max(mar,dim=2).values #keras.layers.GlobalMaxPooling1D()(x)
        output = self.dropout(pooling1) #keras.layers.core.Dropout(0.2)(pooling1)
        if self.outputdim:
            output = self.linear(output) #tf.keras.layers.Dense(outputdim)(output)
            output = torch.sigmoid(output) #keras.layers.Activation("sigmoid")(output)
        return output.squeeze()


class TorchDataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset[0].astype("float32")
        self.Y = dataset[1].astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def trainMarkonv(modelsave_output_prefix,data_set, number_of_kernel, kernel_size,
             random_seed, batch_size, epoch_scheme,GPUID="0", outputName= "Markonv"):

    '''
    Complete BConv training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''


    torch.manual_seed(random_seed)
    random.seed(random_seed)
    mkdir(modelsave_output_prefix+"/"+outputName)
    mkdir(modelsave_output_prefix.replace("result","log")+"/"+outputName)

    modelsave_output_filename = modelsave_output_prefix + "/"+outputName+"/model_KernelNum-" + str(number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_batch_size-" + str(batch_size) + ".pt"
    modellogname = modelsave_output_prefix.replace("result","log")+"/"+outputName+"/"+ "model_KernelNum-" + str(number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_batch_size-" + str(batch_size)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
    # if os.path.exists(test_prediction_output):
    #     print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0
    Trainlossonbatch = []
    validlossonbatch = []
    training_set, test_set = data_set
    train_set, test_set = TorchDataset(training_set), TorchDataset(test_set)
    
    training_set_len = len(training_set[0])
    train_set_len = int(training_set_len*0.8)
    train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_len,training_set_len-train_set_len])
    #
    # print(len(train_set),len(valid_set),len(test_set))

    device = 'cuda:'+GPUID
    net = buildModel(kernel_size, number_of_kernel, 1, outputName)

    net = net.to(device)
    print(net)
    #optimizer = torch.optim.Adadelta(net.parameters(), lr=1, rho=0.9, eps=1.0e-8)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01,alpha=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=30, min_lr=0.001)
    iterations = 0
    best_loss = 100000
    earlS_num = 0
    TrainTime = []
    writer = SummaryWriter(modellogname)
    BCEloss = nn.BCELoss()

    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    for epoch in range(int(epoch_scheme)):
        # 训练
        net.train()
        startTime_for_tictoc = time.time()
        total_loss = 0.0
        tem = 0
        for X_iter, Y_true_iter in train_dataloader:
            X_iter = X_iter.to(device)
            Y_true_iter = Y_true_iter.to(device)
            optimizer.zero_grad()
            Y_pred = net(X_iter)

            loss = BCEloss(Y_pred, Y_true_iter.float())
            loss.backward()
            optimizer.step()

            iterations += 1
            total_loss += loss.item()
            tem = tem + 1
            if iterations % 64 == 0:
                auc_train_batch = roc_auc_score(Y_true_iter.cpu().detach(), Y_pred.cpu().detach())
                loss = loss.item()
                print(f'iterations={iterations}, loss={loss}, auc_train_batch={auc_train_batch}')
        total_loss = total_loss / tem
        Trainlossonbatch.append(total_loss)
        # 验证
        net.eval()
        with torch.no_grad():
            total_loss = 0.0
            tem = 0
            for X_iter,Y_true_iter in valid_dataloader:
                X_iter = X_iter.to(device)
                Y_true_iter = Y_true_iter.to(device)
                Y_pred = net(X_iter)
                loss_iter = BCEloss(Y_pred,Y_true_iter.float())
                total_loss += loss_iter.item()
                tem = tem + 1
            total_loss = total_loss / tem
        validlossonbatch.append(total_loss)
        lr_scheduler.step(total_loss)
        print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
        print(f'valid: epoch={epoch}, loss={total_loss}')
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
            earlS_num = 0
            print("save the best model")
            # print(net.markonv.k_weights)
        else:
            earlS_num = earlS_num+1
        TrainTime.append(time.time() - startTime_for_tictoc)
        if earlS_num >= 50:
            break

    np.save(modellogname, {"trainloss": Trainlossonbatch, "validloss": validlossonbatch})


    return TrainTime




def trainCNN(modelsave_output_prefix, data_set, number_of_kernel, kernel_size,
                 random_seed, batch_size, epoch_scheme, GPUID="0", outputName="CNN"):
    '''
    Complete BConv training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    mkdir(modelsave_output_prefix + "/" + outputName)
    mkdir(modelsave_output_prefix.replace("result", "log") + "/" + outputName)

    modelsave_output_filename = modelsave_output_prefix + "/" + outputName + "/model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_batch_size-" + str(
        batch_size) + ".pt"
    modellogname = modelsave_output_prefix.replace("result", "log") + "/" + outputName + "/" + "model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                   str(kernel_size) + "_seed-" + str(random_seed) + "_batch_size-" + str(batch_size)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
    # if os.path.exists(test_prediction_output):
    #     print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0
    Trainlossonbatch = []
    validlossonbatch = []
    training_set, test_set = data_set
    training_set[0] = np.transpose(training_set[0],[0,2,1])
    test_set[0] = np.transpose(test_set[0],[0,2,1])

    train_set, test_set = TorchDataset(training_set), TorchDataset(test_set)
    training_set_len = len(train_set)
    train_set_len = int(training_set_len*0.8)
    train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_len,training_set_len-train_set_len])
    #
    print(len(train_set),len(valid_set),len(test_set))

    device = 'cuda:' + GPUID
    net = CNN(training_set[0].shape[1],kernel_size, number_of_kernel, 1).to(device)

    print(net)
    # optimizer = torch.optim.Adadelta(net.parameters(), lr=1)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01,alpha=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=30, min_lr=0.001)

    iterations = 0
    best_loss = 100000
    earlS_num = 0

    writer = SummaryWriter(modellogname)
    BCEloss = nn.BCELoss()

    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    TrainTime = []

    for epoch in range(int(epoch_scheme)):
        # 训练
        net.train()
        startTime_for_tictoc = time.time()
        total_loss = 0.0
        tem = 0
        for X_iter, Y_true_iter in train_dataloader:
            X_iter = X_iter.to(device)
            Y_true_iter = Y_true_iter.to(device)
            optimizer.zero_grad()
            Y_pred = net(X_iter)

            loss = BCEloss(Y_pred, Y_true_iter.float())
            loss.backward()
            optimizer.step()

            iterations += 1
            total_loss += loss.item()
            tem = tem + 1
            if iterations % 64 == 0:
                auc_train_batch = roc_auc_score(Y_true_iter.cpu().detach(), Y_pred.cpu().detach())
                loss = loss.item()
                print(f'iterations={iterations}, loss={loss}, auc_train_batch={auc_train_batch}')
        total_loss = total_loss / tem
        Trainlossonbatch.append(total_loss)
        # 验证
        net.eval()
        with torch.no_grad():
            total_loss = 0.0
            tem = 0
            for X_iter, Y_true_iter in valid_dataloader:
                X_iter = X_iter.to(device)
                Y_true_iter = Y_true_iter.to(device)
                Y_pred = net(X_iter)
                loss_iter = BCEloss(Y_pred, Y_true_iter.float())
                total_loss += loss_iter.item()
                tem = tem + 1
            total_loss = total_loss / tem
        validlossonbatch.append(total_loss)
        lr_scheduler.step(total_loss)
        print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
        print(f'valid: epoch={epoch}, loss={total_loss}')
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
            earlS_num = 0
            print("save the best model")
            # print(net.markonv.k_weights)
        else:
            earlS_num = earlS_num + 1

        TrainTime.append(time.time() - startTime_for_tictoc)

        if earlS_num >= 50:
            break

    np.save(modellogname, {"trainloss": Trainlossonbatch, "validloss": validlossonbatch})

    return TrainTime


def TransforTraining(onehotarray):
    """

    Args:
        onehotarray:

    Returns:

    """
    outputarray = np.zeros((onehotarray.shape[0],onehotarray.shape[1]*onehotarray.shape[1]
                            ,onehotarray.shape[2]-1))

    for i in range(onehotarray.shape[0]):
        for k in range(onehotarray.shape[2]-1):
            for j1 in range(onehotarray.shape[1]):
                for j2 in range(onehotarray.shape[1]):
                    outputarray[i,(j1)*onehotarray.shape[1]+j2,k] = onehotarray[i,j1,k]*onehotarray[i,j2,k+1]

    return outputarray




def trainHOCNNLB(modelsave_output_prefix, data_set, number_of_kernel, kernel_size,
                 random_seed, batch_size, epoch_scheme, GPUID="0", outputName="CNN"):
    '''
    Complete BConv training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    mkdir(modelsave_output_prefix + "/" + outputName)
    mkdir(modelsave_output_prefix.replace("result", "log") + "/" + outputName)

    modelsave_output_filename = modelsave_output_prefix + "/" + outputName + "/model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_batch_size-" + str(
        batch_size) + ".pt"
    modellogname = modelsave_output_prefix.replace("result", "log") + "/" + outputName + "/" + "model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                   str(kernel_size) + "_seed-" + str(random_seed) + "_batch_size-" + str(batch_size)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
    # if os.path.exists(test_prediction_output):
    #     print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0

    training_set, test_set = data_set
    training_set[0] = TransforTraining(np.transpose(training_set[0],[0,2,1]))
    test_set[0] = TransforTraining(np.transpose(test_set[0],[0,2,1]))

    train_set, test_set = TorchDataset(training_set), TorchDataset(test_set)
    training_set_len = len(train_set)
    train_set_len = int(training_set_len*0.8)
    train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_len,training_set_len-train_set_len])
    #
    print(len(train_set),len(valid_set),len(test_set))

    device = 'cuda:' + GPUID
    net = HOCNNLB(training_set[0].shape[1],kernel_size, number_of_kernel, 1).to(device)

    print(net)
    # optimizer = torch.optim.Adadelta(net.parameters(), lr=1)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01,alpha=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=30, min_lr=0.001)

    iterations = 0
    best_loss = 100000
    earlS_num = 0

    BCEloss = nn.BCELoss()

    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    TrainTime = []
    Trainlossonbatch = []
    validlossonbatch = []

    for epoch in range(int(epoch_scheme)):
        # 训练
        net.train()
        startTime_for_tictoc = time.time()
        total_loss = 0.0
        tem = 0
        for X_iter, Y_true_iter in train_dataloader:
            X_iter = X_iter.to(device)
            Y_true_iter = Y_true_iter.to(device)
            optimizer.zero_grad()
            Y_pred = net(X_iter)

            loss = BCEloss(Y_pred, Y_true_iter.float())
            loss.backward()
            optimizer.step()

            iterations += 1
            total_loss += loss.item()
            tem = tem + 1
            if iterations % 64 == 0:
                auc_train_batch = roc_auc_score(Y_true_iter.cpu().detach(), Y_pred.cpu().detach())
                loss = loss.item()
                print(f'iterations={iterations}, loss={loss}, auc_train_batch={auc_train_batch}')
        total_loss = total_loss / tem
        Trainlossonbatch.append(total_loss)
        # 验证
        net.eval()
        with torch.no_grad():
            total_loss = 0.0
            tem = 0
            for X_iter, Y_true_iter in valid_dataloader:
                X_iter = X_iter.to(device)
                Y_true_iter = Y_true_iter.to(device)
                Y_pred = net(X_iter)
                loss_iter = BCEloss(Y_pred, Y_true_iter.float())
                total_loss += loss_iter.item()
                tem = tem+1
            total_loss=total_loss/tem
        validlossonbatch.append(total_loss)
        lr_scheduler.step(total_loss)
        print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
        print(f'valid: epoch={epoch}, loss={total_loss}')
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
            earlS_num = 0
            print("save the best model")
            # print(net.markonv.k_weights)
        else:
            earlS_num = earlS_num + 1

        TrainTime.append(time.time() - startTime_for_tictoc)

        if earlS_num >= 50:
            break

    np.save(modellogname,{"trainloss":Trainlossonbatch,"validloss":validlossonbatch})

    return TrainTime
