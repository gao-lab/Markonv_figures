import numpy as np
import pdb
import os
import h5py
import random
from sklearn.model_selection import StratifiedKFold
import math
import time
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
plt.switch_backend('agg')

def GenerateMTM(path, lenlist, seed):
    """
    generate Marcov Transition matrix
    
    :return:
    """
    complexityChoice=[]
    for i in range(len(lenlist)):
        f = h5py.File(path+"simulationMTM"+ str(lenlist[i])+"_"+str(seed)+"_MatrixT.hdf5", "w")
        MatrixT = np.random.randint(0, 100, (4, 4, 10))
        pos = np.argsort(MatrixT, axis=1)
        initProbT = np.random.randint(0, 100, 4)
        initPos = np.argsort(initProbT)
        InitProb = np.zeros(4)
        InitProb[initPos[0]]=0.4
        InitProb[initPos[1]]=0.4
        InitProb[initPos[2]]=0.1
        InitProb[initPos[3]]=0.1
        f.create_dataset("pos",data=pos)
        f.create_dataset("InitProb",data=InitProb)
        f.close()

        # One choice can lead to one choice
        complexity = 1
        mkdir(path+str(complexity))
        f1 = h5py.File(path+str(complexity)+"/simulationMTM"+ str(lenlist[i])+"_"+str(seed)+"_1.hdf5", "w")
        Matrix = np.zeros((4, 4, lenlist[i]))
        Matrix[:,:,0] = InitProb.reshape((1,4,1))
        for position in range(4):
            for k in range(1,lenlist[i]):
                Matrix[position,pos[position,3, k], k] = 1
        f1.create_dataset(str(lenlist[i])+ "_" +str(i), data=Matrix)
        f1.close()
        complexityChoice.append(1)

        # One choice can lead to two choices
        complexity = 291
        mkdir(path+str(complexity))
        f1 = h5py.File(path+str(complexity)+"/simulationMTM"+ str(lenlist[i])+"_"+str(seed)+"_291.hdf5", "w")
        Matrix = np.zeros((4, 4, lenlist[i]))
        Matrix[:,:,0] = InitProb.reshape((1,4,1))
        for position in range(4):
            for k in range(1,lenlist[i]):
                Matrix[position,pos[position,3, k], k] = 0.9
                Matrix[position,pos[position,0, k], k] = 0.1
        f1.create_dataset(str(lenlist[i])+ "_" +str(i), data=Matrix)
        f1.close()
        complexityChoice.append(291)

        complexity = 282
        mkdir(path+str(complexity))
        f1 = h5py.File(path+str(complexity)+"/simulationMTM"+ str(lenlist[i])+"_"+str(seed)+"_282.hdf5", "w")
        Matrix = np.zeros((4, 4, lenlist[i]))
        Matrix[:,:,0] = InitProb.reshape((1,4,1))
        for position in range(4):
            for k in range(1,lenlist[i]):
                Matrix[position,pos[position,3, k], k] = 0.8
                Matrix[position,pos[position,0, k], k] = 0.2
        f1.create_dataset(str(lenlist[i])+ "_" +str(i), data=Matrix)
        f1.close()
        complexityChoice.append(282)

        complexity = 273
        mkdir(path+str(complexity))
        f1 = h5py.File(path+str(complexity)+"/simulationMTM"+ str(lenlist[i])+"_"+str(seed)+"_273.hdf5", "w")
        Matrix = np.zeros((4, 4, lenlist[i]))
        Matrix[:,:,0] = InitProb.reshape((1,4,1))
        for position in range(4):
            for k in range(1,lenlist[i]):
                Matrix[position,pos[position,3, k], k] = 0.7
                Matrix[position,pos[position,0, k], k] = 0.3
        f1.create_dataset(str(lenlist[i])+ "_" +str(i), data=Matrix)
        f1.close()
        complexityChoice.append(273)

    return complexityChoice


def entropy(MTM, i, k, mid):
    """ Calculate entropy of MM
    """
    sum = 0
    if k == MTM.shape[-1] - 1:
        for j in range(4):
            p = mid * MTM[i, j, k]
            if p != 0:
                sum += p * math.log2(p)
        return sum
    for j in range(4):
        sum += entropy(MTM, j, k + 1, mid * MTM[i, j, k])
    return sum


def visualize(path, lenlist, seed, complexityChoice):
    """
    Calculate entropy and plot motif one by one.
    """
    calentropy=True
    if calentropy:
        with open(path+'motif/reportEntropy.txt', 'w') as f:
            f.write('complexity\tentropy\n')
        compChoice = []
        compEntropy = []
    mkdir(f"{path}motif")
    for complexity in complexityChoice:
        f = h5py.File(
            path +str(complexity)+ "/simulationMTM" +str(lenlist[0])+"_"+ str(seed) + "_" + str(complexity) + ".hdf5",
            'r')

        for key in f.keys():
            sum = 0
            MTM = f[key]
            kernel_len = 10

            fig, axes = plt.subplots(1, kernel_len, figsize=(2*kernel_len, 2))
            for k in range(kernel_len):
                ax = axes[k]
                if k == 0:
                    sns.heatmap(MTM[:, 0, k].reshape(4, 1),
                                ax=ax,
                                annot=True,
                                vmin=0,
                                vmax=1,
                                cmap=sns.color_palette("light:#5A9", as_cmap=True),
                                cbar=False,
                                xticklabels=[" "],
                                yticklabels=["A","C","G","T"],
                                linewidths=0)
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    ax.set_title("S1",fontsize=15)
                else:
                    sns.heatmap(MTM[:, :, k].reshape(4, 4),
                                ax=ax,
                                annot=True,
                                vmin=0,
                                vmax=1,
                                cmap=sns.color_palette("light:#5A9", as_cmap=True),
                                cbar=False,
                                xticklabels=["A","C","G","T"],
                                yticklabels=False,
                                linewidths=0)
                    ax.set_title(f"S{k} → S{k+1}",fontsize=15)
                ax.xaxis.tick_top()
                ax.tick_params(bottom=False,top=False,left=False,right=False)

            plt.savefig(f'{path}motif/heatmap_{complexity}.png', bbox_inches='tight')
            plt.close()

            if calentropy:
                for i in range(4):
                    sum += entropy(MTM, i, 1, MTM[i, 0, 0])
                compChoice.append(complexity)
                compEntropy.append(-sum)
    if calentropy:
        with open(path+'motif/reportEntropy.txt', 'a') as f:
            f.write('\n'.join([f'{complexity}\t{entropy}'
                    for complexity, entropy in zip(compChoice, compEntropy)]))



def drawMotif():
    """
    Draw all motifs together.
    """
    kernel_len = 10

    # Get calculated entropy
    compEntropy = pd.read_csv(path+'motif/reportEntropy.txt',sep='\t',index_col='complexity')

    fig, axes = plt.subplots(len(complexityChoice),kernel_len,figsize=(20,2.4*len(complexityChoice)))
    for i,complexity in enumerate(complexityChoice):
        f = h5py.File(
            path + str(complexity)+"/simulationMTM" +str(lenlist[0])+"_"+ str(seed) + "_" + str(complexity) + ".hdf5",
            'r')
        entropy = compEntropy.loc[complexity,"entropy"]

        for key in f.keys():
            MTM = f[key]

            for k in range(kernel_len):
                ax = axes[i, k]
                if k == 0:
                    sns.heatmap(MTM[:, 0, k].reshape(4, 1),
                                ax=ax,
                                annot=True,
                                vmin=0,
                                vmax=1,
                                cmap=sns.color_palette("light:#5A9", as_cmap=True),
                                cbar=False,
                                xticklabels=[" "],
                                yticklabels=["A","C","G","T"],
                                linewidths=0)
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    if i == 0:
                        ax.set_title("S1", fontsize=18)
                else:
                    if i == 0:
                        sns.heatmap(MTM[:, :, k].reshape(4, 4),
                                    ax=ax,
                                    annot=True,
                                    vmin=0,
                                    vmax=1,
                                    cmap=sns.color_palette("light:#5A9", as_cmap=True),
                                    cbar=False,
                                    xticklabels=["A","C","G","T"],
                                    yticklabels=False,
                                    linewidths=0)
                        ax.set_title(f"S{k} → S{k+1}", fontsize=18)

                    else:
                        sns.heatmap(MTM[:, :, k].reshape(4, 4),
                                    ax=ax,
                                    annot=True,
                                    vmin=0,
                                    vmax=1,
                                    cmap=sns.color_palette("light:#5A9", as_cmap=True),
                                    cbar=False,
                                    xticklabels=False,
                                    yticklabels=False,
                                    linewidths=0)
                ax.xaxis.tick_top()
                ax.tick_params(bottom=False,top=False,left=False,right=False)
            if i == 0:
                axes[i,5].set_title(f"Motif {i+1} (entropy={entropy:.2f})\nS5 → S6", fontsize=18)
            else:
                axes[i,5].set_title(f"Motif {i+1} (entropy={entropy:.2f})", fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{path}motif/heatmap_all_motifs.png', bbox_inches='tight')
    plt.close()


def motifInMatrix(MTMlist, AllseqArray, seqLen):
    """

    :param MTMlist:
    :param AllseqArray: [n,len,4]
    :param seqLen:
    :return:
    """
    
    def MTMToMotif(MTM):
        """
        Generate a fragment based on the probability of the motif
        :param MTM:
        :return:
        """
        motif = np.zeros((MTM.shape[-1],4), dtype=np.int)
        
        randomarray = np.zeros((10000,), dtype=np.int)
        Anum = int(MTM[0,0, 0] * 10000)
        Cnum = int(MTM[1,0, 0] * 10000) + Anum
        Gnum = int(MTM[2,0, 0] * 10000) + Cnum
        Tnum = int(MTM[3,0, 0] * 10000) + Gnum
        randomarray[Anum:Cnum] = 1
        randomarray[Cnum:Gnum] = 2
        randomarray[Gnum:Tnum] = 3
        out = random.sample(list(randomarray), 1)[0]
        motif[0, out] = 1
        
        for i in range(1, MTM.shape[-1]):
            randomarray = np.zeros((10000,), dtype=np.int)
            Anum = int(MTM[out,0,i] * 10000)
            Cnum = int(MTM[out,1,i] * 10000) + Anum
            Gnum = int(MTM[out,2,i] * 10000) + Cnum
            Tnum = int(MTM[out,3,i] * 10000) + Gnum
            randomarray[Anum:Cnum] = 1
            randomarray[Cnum:Gnum] = 2
            randomarray[Gnum:Tnum] = 3
            out = random.sample(list(randomarray), 1)[0]
            motif[i, out] = 1
            
        return motif
    
    seqnum = AllseqArray.shape[0]
    InsertPlace = np.random.randint(0, seqLen - 20, [seqnum, ])
    InsertMotifType = np.random.randint(0, len(MTMlist), [seqnum, ])
    for i in range(seqnum):
        MTM = MTMlist[InsertMotifType[i]]
        insertPlaceTem = InsertPlace[i]
        AllseqArray[i, insertPlaceTem:(insertPlaceTem + MTM.shape[-1]), :] = MTMToMotif(MTM)
    
    return AllseqArray, InsertMotifType


def GeneRateMatrix(path, hyper,outPath,complexity):
    """
    Generate and store dataset after generating motifs.
    :param path:
    :return:
    """
    number = hyper["number"]
    length = hyper["length"]
    
    f = h5py.File(path, 'r')
    MTMlist = []

    for key in f.keys():
        MTMlist.append(f[key])
        

    def GenerateRandomMatrix(seqNum, seqLen):
        def seqSeriesToMatrix(seqSeries, seqLen):
            seqMatrix = np.zeros([seqLen, 4])
            for i in range(seqLen):
                seqMatrix[i, seqSeries[i]] = 1
            return seqMatrix
        
        Allseq = np.random.randint(0, 4, [seqNum, seqLen])
        AllseqArray = np.zeros([seqNum, seqLen, 4])
        for i in range(seqNum):
            seqTem = Allseq[i, :]
            seqTemMatrix = seqSeriesToMatrix(seqTem, seqLen)
            AllseqArray[i, :, :] = seqTemMatrix
        
        return AllseqArray
    
    InitMatrixPos = GenerateRandomMatrix(int(number/2), length)
    InitMatrixAll = GenerateRandomMatrix(number, length)
    label = np.zeros((number,))
    InsertMotifTypeAll =  np.zeros((number,))
    seq_pos_matrix_out, InsertMotifType = motifInMatrix(MTMlist, InitMatrixPos, length)

    InitMatrixAll[:int(number/2),:,:] = seq_pos_matrix_out
    label[:int(number/2)] = 1
    InsertMotifTypeAll[:int(number/2)] = InsertMotifType
    
    index_shuffle = list(range(number))
    random.shuffle(index_shuffle)
    seq_matrix_out = InitMatrixAll[index_shuffle, :, :]
    
    label_out = label[index_shuffle]
    id_out = InsertMotifTypeAll[index_shuffle].astype("string_")
    outData = [seq_matrix_out, label_out, id_out]
    StoreTrainSet(rootPath=outPath, allData=outData,complexity=complexity)
    

    
def StoreTrainSet(rootPath, allData,complexity,ValNum=10, RandomSeeds=233):
    """
    store different dataset
    :param rootPath:
    :param ValNum: all data size /test size
    :param RandomSeeds: for generating testing dataset
    :param allData: All data
    """
    dataNum = allData[1].shape[0]
    split_train_index_and_test_index_list = cross_validation(number_of_folds=ValNum, total_number=dataNum, random_seeds=RandomSeeds)
    i=0
    outDataTem = split_dataset(split_train_index_and_test_index_list, fold=i, data_x=allData[0], data_y=allData[1], data_id=allData[2])

    mkdir(rootPath)
    mkdir(rootPath+str(complexity))
    training_path = rootPath +str(complexity)+ "/train.hdf5"
    test_path = rootPath +str(complexity)+"/test.hdf5"

    f_train = h5py.File(training_path, "w")
    f_test = h5py.File(test_path, "w")

    f_train.create_dataset("sequences",data = outDataTem[0])
    f_train.create_dataset("labs",data=outDataTem[1])
    f_train.create_dataset("seq_idx",data=outDataTem[2])
    f_train.close()
    f_test.create_dataset("sequences",data=outDataTem[3])
    f_test.create_dataset("labs",data=outDataTem[4])
    f_test.create_dataset("seq_idx",data=outDataTem[5])
    f_test.close()


def cross_validation(number_of_folds, total_number, random_seeds=233):
    """
    :param number_of_folds:
    :param total_number:
    :param random_seeds:
    :return:
    """
    x = np.zeros((total_number, ), dtype=np.int)
    split_iterator = StratifiedKFold(n_splits=number_of_folds, random_state=random_seeds, shuffle=True)
    split_train_index_and_test_index_list = [
        (train_index, test_index)
        for train_index, test_index in split_iterator.split(x,x)
    ]
    return(split_train_index_and_test_index_list)

def split_dataset(split_index_list, fold, data_x, data_y, data_id=None):
    """
    generate training dataset and test data set
    :param split_index_list:
    :param fold:
    :param data_id:
    :param data_x:X
    :param data_y:Y
    :return:
    """
    id_train=data_id[split_index_list[fold][0].tolist()]
    x_train=data_x[split_index_list[fold][0].tolist()]
    y_train=data_y[split_index_list[fold][0].tolist()]
    id_test=data_id[split_index_list[fold][1].tolist()]
    x_test=data_x[split_index_list[fold][1].tolist()]
    y_test=data_y[split_index_list[fold][1].tolist()]
    return [x_train, y_train, id_train, x_test, y_test, id_test]

 
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

if __name__ == '__main__':
    start=time.time()
    lenlist = [10]
    path = "./"
    seed=20
    random.seed(seed)
    np.random.seed(seed)
    mkdir(path)
    mkdir(path+'motif')

    complexityChoice = GenerateMTM(path,lenlist,seed)
    print('Generate MTM done. total time:',time.time()-start)

    complexityChoice = [1,291,282,273]
    visualize(path, lenlist, seed, complexityChoice)
    print('Visualize done. total time:',time.time()-start)

    hyper = {"number":6000,
        "length":1000
    }
    for complexity in complexityChoice:
        GeneRateMatrix(path+str(complexity)+"/simulationMTM"+ str(lenlist[0])+"_"+str(seed)+"_"+str(complexity) +".hdf5",hyper,path, complexity)
    print('Generate sequence done. total time:', time.time()-start)

    # Draw all motif on a single figure.
    drawMotif()

