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

    data_list = glob.glob(path + "/*")
    data_list.sort()
    auclist = []
    outputcsv = {"DataName":[],"AUROC":[]}
    for DataName in data_list:
        DataName = DataName + "/" + Mtype
        filelist = glob.glob(DataName + "/Report*")
        if len(filelist) > 0:
            auc = 0

            for file in filelist:
                with open(file, "rb") as f:
                    ret = (pickle.load(f)).tolist()
                    if ret["test_auc"] > auc:
                        auc = ret["test_auc"]
                        tmp_path = file.replace("pkl", "hdf5")
                        BestModel = tmp_path.replace("/Report_KernelNum-", "/model_KernelNum-")
            auclist.append(auc)
            print(BestModel)
            print(auc)
            f = open(DataName+"/bestmodel.txt",mode="w")
            f.write(BestModel)
            outputcsv["DataName"].append(DataName.split("/")[-2])
            outputcsv["AUROC"].append(auc)


    return auclist


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
    Pddict.to_csv("./test.csv")
    cols = Pddict.columns.tolist()
    cols = cols[1:2] + cols[0:1]+ cols[2:]
    cols = ['Markonv- \n based network', "Markonv-based \n HOCNNLB", 'HOCNNLB-1', 'HOCNNLB-2', 'HOCNNLB-3','HOCNNLB-4',]
    cols = ['Markonv- \n based network',  'HOCNNLB-1', 'HOCNNLB-2', 'HOCNNLB-3','HOCNNLB-4',"iDeepS","DeepBind"]
    # Pddict =Pddict.drop("VCNN-based model",axis=1)

    # Pddict.drop()
    Pddict = Pddict[cols]
    plt.figure(figsize=(10, 8))
    plt.rc('font', family='Times New Roman')
    ax = sns.boxplot(data=Pddict,saturation=0.4)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0,fontsize= 13)
    plt.ylim(0.5, 1.09)
    plt.ylabel("AUROC",fontsize= 20)
    plt.xlabel("networks",fontsize= 20)
    plt.title("AUROC distribution across all 31 RBP datasets", fontsize=25)
    # Pddict.boxplot()

    pairs = [[0,1],[0,2]]
    for i in range(len(pairs)):
        pair = pairs[i]
        x1, x2 = pair  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = 1.01+0.02*(i), 0.01*(i+1), 'k'
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        pvalue = scipy.stats.wilcoxon(Pddict[cols[pair[0]]], Pddict[cols[pair[1]]])[1]
        if pvalue<0.001:
            label = '***'
        else:
            label = round(pvalue,4)
        plt.text((x1 + x2) * .5, y + h, str(label), ha='center', va='bottom', color=col)

    plt.tight_layout()

    plt.savefig(path + "auc.png", dpi=400)
    plt.close('all')

    # draw scatter
    # plt.scatter(ModeltypeDict["BConv-based model"],ModeltypeDict["CNN-based model"])
    # plt.plot([0.5,1],[0.5,1])
    # plt.ylim(0.5, 1)
    # plt.ylabel("CNN-based model")
    # plt.xlabel("BConv-based model")  # 我们设置横纵坐标的标题。
    # # Pddict.boxplot()
    # plt.savefig(path + "scatter.png")
    # plt.close('all')

    # draw scatter
    # plt.scatter(ModeltypeDict["Markonv-based model"], ModeltypeDict["HOCNNLB"])
    # plt.plot([0.5, 1], [0.5, 1])
    # plt.ylim(0.5, 1)
    # plt.ylabel("HOCNNLB")
    # plt.xlabel("BCNN-based model")  # 我们设置横纵坐标的标题。
    # # Pddict.boxplot()
    # plt.savefig(path + "HOCNNvsMarkonv.png")
    # plt.close('all')


#
def Main():
    """

    :return:
    """

    path = "../../result/HOCNNLB/HOCNNLB/"
    ModelType = ["MarkonvV", "HOCNNLB"]


    ResultDict = {}
    length = 3000

    for Mtype in ModelType:
        auclist = loadResult(path, Mtype)
        if Mtype == "MarkonvV":
            Mtype = "Markonv" + "- \n based network"
        elif Mtype == "HOCNNLB":
            Mtype = "Markonv-based \n HOCNNLB"
        ResultDict[Mtype] = auclist
        length = min(length, len(auclist))
    for key in ResultDict.keys():
        ResultDict[key] = ResultDict[key][:length]
    print(length)
    # for i in range(length):
    #     if ResultDict["BCNN"][i] - ResultDict["CNN"][i]>0.1:
    #         print(i, ResultDict["BCNN"][i] - ResultDict["CNN"][i])
    ###old model result###
    path = "./result.csv"

    dict2 = loadideep(path)

    ### 顺序不同
    ResultDict = dict(ResultDict, **dict2)
    Draw("../../result//HOCNNLB/", ResultDict)


if __name__ == '__main__':
    Main()

    # import glob
    # import os
    # path = "/rd2/lijy/BayesianCNN/result/RBP/Hdf5/"
    #
    # data_list = glob.glob(path + "/*")
    # data_list.sort()
    # auclist = []
    # for DataName in data_list:
    #
    #     Outpath = DataName + "/" + "VCNN/"
    #
    #     # cmd = "mkdir " + Outpath
    #     # os.system(cmd)
    #     cmd2 = "mv " + DataName+"/model_Kernel* " + Outpath
    #     cmd3 = "mv " + DataName+"/Report* " + Outpath
    #     os.system(cmd2)
    #     os.system(cmd3)


