# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import pandas as pd
import seaborn as sns
import pdb
plt.switch_backend('agg')
import copy



def loadResultpkl(path):
    """

    :param path:
    :param modelType:
    :return: auc list
    """

    filelist = glob.glob(path+"/Report*")
    losslist = []
    f = open(path + "/bestmodel.txt","r")
    BestModel = f.readlines()[0]
    f.close()
    for file in filelist:
        tmp_path = file.replace("pkl", "hdf5")
        Model = tmp_path.replace("/Report_KernelNum-", "/model_KernelNum-")
        if Model == BestModel:
            with open(file, "rb") as f:
                ret = (pickle.load(f)).tolist()
                pdb.set_trace()
                losslist = ret["auc"][0]

    return losslist

def DrawPic(lossDict, output_path, dataname, patience=50):
    """

    Args:
        lossDict:
        output_path:
        dataname:
        patience:
    Returns:

    """

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    color = {"Markonv-based network":"silver", "convolution-based network":"lightcoral",
    "bonito":"lightcoral","Markonv-based basecaller":"silver"}

    validationColor = {"convolution-based network":"mediumslateblue","Markonv-based network":"aquamarine"}

    for mode in lossDict.keys():
        print(mode)

        modeltype = mode
        trainloss = lossDict[mode]["trainloss"][:len(lossDict[mode]["trainloss"])]
        valloss = lossDict[mode]["validloss"][:len(lossDict[mode]["validloss"])]

        ax.plot(range(len(trainloss)),trainloss,c=color[mode],label=modeltype+" train loss")
        ax.plot(range(len(valloss)),valloss,c=color[mode],linestyle="--",label=modeltype+" valid loss")
        ax.plot([len(valloss)-patience-1,len(valloss)-patience-1],[np.min(valloss)-0.2,np.max(valloss)+0.2],c=validationColor[mode], linestyle='dashdot')
    plt.title("Dataset "+dataname, fontsize='20')
    plt.ylabel("loss", fontsize='15')
    plt.xlabel("Epoch", fontsize='15')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(prop={'size': 12})
    # plt.show()
    plt.tight_layout()
    plt.savefig(output_path+".png",dpi=400)
    plt.close()
    print(dataname)

def mkdir(path):
    """
    Create a directory
    :param path: Directory path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def loadbonitoresult():
    """

    :return:
    """
    MarkonvLosspath = "../../result/bonito/training_history/markonv.csv"
    BonitoLosspath = "../../result/bonito/training_history/conv.csv"
    lossdict = {}
    Markonvfile = pd.read_csv(MarkonvLosspath)
    bonitofile = pd.read_csv(BonitoLosspath)

    lossdict["Markonv-based basecaller"] = {"trainloss":Markonvfile["train_loss"],"validloss":Markonvfile["validation_loss"]}
    lossdict["bonito"] = {"trainloss":bonitofile["train_loss"],"validloss":bonitofile["validation_loss"]}
    return lossdict


def main():

    modeltype = ["CNN", "Markonv"]
    outputName = {"CNN":"convolution-based network", "BConv":"Markonv-tf", "Markonv":"Markonv-based network"}
    datasetnamelist = [1,291,282,273]
    randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927, 42, 48, 100, 420, 60, 320, 7767, 51]

    datasetnamedict = {"1":"1","291":"2","282":"3","273":"4"}
    result_path = "../../log/SpeedCompare/"
    output_path = "../../result/SpeedCompare/figure/"
    mkdir(output_path)
    for dataname in datasetnamelist:
        lossDict = {}
        for randomseed in randomSeedslist:
            for mode in modeltype:
                path = result_path+str(dataname)+"/"+mode

                losslist = np.load(glob.glob(path+"/*"+str(randomseed)+"*npy")[0], allow_pickle=True).item()
                lossDict[outputName[mode]] = losslist

            DrawPic(lossDict, output_path+str(datasetnamedict[str(dataname)])+"_"+str(randomseed), str(datasetnamedict[str(dataname)]),patience=50)


    ### training loss on bonito
    lossDict = loadbonitoresult()

    DrawPic(lossDict, output_path, "Oxford nanopore basecalling",patience=0)


if __name__ == '__main__':
    main()