# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import pandas as pd
import seaborn as sns
import pdb
import scipy
plt.switch_backend('agg')


def loadResult(path, Mtype):
    """

    :param path:
    :param modelType:
    :return: auc list
    """

    data_list = glob.glob(path + "/HOCNNLB/*")
    data_list.sort()
    outputcsv = {}
    for DataName in data_list:
        DataName = DataName + "/" + Mtype
        filelist = glob.glob(DataName + "/Report*")
        name = DataName.split("/")[-2]
        outputcsv[name] = []
        if len(filelist) > 0:
            auc = 0

            for file in filelist:
                with open(file, "rb") as f:
                    ret = (pickle.load(f)).tolist()
                    outputcsv[name].append(ret["test_auc"])
    return outputcsv


def loadideep(file):
    """

    :param file:
    :return:
    """
    f = pd.read_csv(file)

    return f


def Draw(path, ModeltypeDict):
    """

    :param path:
    :param ModeltypeDict:
    :return:
    """

    # draw box plot

    Pddict = pd.DataFrame(ModeltypeDict)
    plt.figure(figsize=(10, 15))
    plt.rc('font', family='Times New Roman')
    ax = sns.boxplot(data=Pddict,saturation=0.4,orient="h")
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=5 , rotation=0)
    plt.xlim(0.5, 1)
    plt.ylabel("RBP names",fontsize= 15)
    plt.xlabel("AUROC",fontsize= 15)
    plt.title("AUROC distribution across all 31 RBP datasets", fontsize=15)
    # Pddict.boxplot()


    plt.tight_layout()

    plt.savefig(path + "auc.png", dpi=400)
    plt.close('all')


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

def sta(result, outputpath):
    """

    """

    stacsv = {"Dataname":[],"Mean value":[],"Max value":[],"Min value":[],"Standard deviation":[]}
    mkdir(outputpath)
    for key in result.keys():

        stacsv["Dataname"].append(key)
        stacsv["Mean value"].append(np.mean(result[key]))
        stacsv["Max value"].append(np.max(result[key]))
        stacsv["Min value"].append(np.min(result[key]))
        stacsv["Standard deviation"].append(np.std(result[key]))

    stacsv = pd.DataFrame(stacsv)
    stacsv.to_csv(outputpath+"sta.csv")


#
def Main():
    """

    :return:
    """

    path = "../../result/HOCNNLB/"
    Mtype = "MarkonvV"


    result = loadResult(path, Mtype)
    sta(result, outputpath=path+"roboust/")
    ###old model result###

    ### 顺序不同
    Draw(path+"roboust/", result)


if __name__ == '__main__':
    Main()


