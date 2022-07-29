import pandas as pd
import numpy as np
import glob
import os
import pdb
import random
import h5py
from sklearn.model_selection import StratifiedKFold

from ushuffle import Shuffler, shuffle


def seq_to_matrix(seq, seq_matrix, seq_order):
    '''
    change target 3D tensor according to sequence and order
    :param seq: Input single root sequence
    :param seq_matrix: Input initialized matrix
    :param seq_order: This is the first sequence
    :return:
    '''
    for i in range(len(seq)):
        if ((seq[i] == 'A') | (seq[i] == 'a')):
            seq_matrix[seq_order, i, 0] = 1
        if ((seq[i] == 'C') | (seq[i] == 'c')):
            seq_matrix[seq_order, i, 1] = 1
        if ((seq[i] == 'G') | (seq[i] == 'g')):
            seq_matrix[seq_order, i, 2] = 1
        if ((seq[i] == 'T') | (seq[i] == 't')):
            seq_matrix[seq_order, i, 3] = 1
    return seq_matrix

def k_mer_shuffle(seq_shape, seq_series, k=2):
    """
    Return a shuffled negative matrix
    :param seq_series:sequence list
    :param seq_shape:
    :param k:kshuffle
    :return:
    """
    seq_shuffle_matrix = np.zeros(seq_shape)

    for i in range(seq_shape[0]):
        seq = seq_series[i]
        shuffler = Shuffler(seq, k)
        seqres = shuffler.shuffle()
        seq_shuffle_matrix = seq_to_matrix(seqres, seq_shuffle_matrix, i)

    return seq_shuffle_matrix

def LoadMotif(path):
    motifTxt = glob.glob(path+"/*.txt")
    motiflist = []
    for file in motifTxt:
        temMotif = np.loadtxt(file)
        motiflist.append(temMotif)
    return motiflist

def GenerateRandomMatrix(seqNum,seqLen):
    def seqSeriesToMatrix(seqSeries,seqLen):
        seqMatrix = np.zeros([seqLen, 4])
        for i in range(seqLen):
            seqMatrix[i,seqSeries[i]] =1
        return seqMatrix

    Allseq = np.random.randint(0,4,[seqNum,seqLen])
    AllseqArray = np.zeros([seqNum,seqLen,4])
    for i in range(seqNum):
        seqTem = Allseq[i,:]
        seqTemMatrix = seqSeriesToMatrix(seqTem,seqLen)
        AllseqArray[i, :, :] = seqTemMatrix

    return AllseqArray


def motifInMatrix(motiflist, AllseqArray,seqLen):
    """

    :param motiflist:
    :param AllseqArray: [n,len,4]
    :param seqLen:
    :return:
    """
    def pwmToMotif(motifPwm):
        """
        Generate a fragment based on the probability of the motif
        :param motifPwm:
        :return:
        """
        motif = np.zeros(motifPwm.shape, dtype=np.int)
        for i in range(motifPwm.shape[0]):
            randomarray = np.zeros((10000,), dtype=np.int)
            Anum = int(motifPwm[i,0]*10000)
            Cnum = int(motifPwm[i,1]*10000) + Anum
            Gnum = int(motifPwm[i,2]*10000) + Cnum
            Tnum = int(motifPwm[i,3]*10000) + Gnum
            randomarray[Anum:Cnum] = 1
            randomarray[Cnum:Gnum] = 2
            randomarray[Gnum:Tnum] = 3
            out = random.sample(list(randomarray),1)[0]
            motif[i, out] = 1
        
        return motif
    
    seqnum = AllseqArray.shape[0]
    InsertPlace = np.random.randint(0, seqLen-20,[seqnum,])
    InsertMotifType = np.random.randint(0, len(motiflist),[seqnum,])
    for i in range(seqnum):
        motif = motiflist[InsertMotifType[i]]
        insertPlaceTem = InsertPlace[i]
        AllseqArray[i,insertPlaceTem:(insertPlaceTem+motif.shape[0]),:] = pwmToMotif(motif)

        
    return AllseqArray,InsertMotifType

def mkdir(path):
    """

    :param path:
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False


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


def StoreTrainSet(rootPath, allData,ValNum=10, RandomSeeds=233):
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
    training_path = rootPath + "/train.hdf5"
    test_path = rootPath + "/test.hdf5"

    f_train = h5py.File(training_path)
    f_test = h5py.File(test_path)

    f_train.create_dataset("sequences",data = outDataTem[0])
    f_train.create_dataset("labs",data=outDataTem[1])
    f_train.create_dataset("seq_idx",data=outDataTem[2])
    f_train.close()
    f_test.create_dataset("sequences",data=outDataTem[3])
    f_test.create_dataset("labs",data=outDataTem[4])
    f_test.create_dataset("seq_idx",data=outDataTem[5])
    f_test.close()

def MatrixToSeq(Matrix):
    """

    """
    seqlist = []
    voca = ["A", "C", "G", "T"]


    for i in range(Matrix.shape[0]):
        seqMatrix = Matrix[i]
        seq = ""
        for j in range(seqMatrix.shape[0]):
            pos = np.where(seqMatrix[j]==1)[0][0]
            seq = seq+voca[pos]
        seqlist.append(seq)

    return seqlist


def GenerateAndSaveData(output_dir, DataDir,seqNum, seqLen):
    """
    Generate And Save Data
    :param output_dir: output hdf5 path
    :param DataDir: input data path
    :param seqNum:
    :param seqLen:
    :return:
    """

    Motiflist = LoadMotif(DataDir)
    # GeneRate Positive Dataset
    InitMatrix = GenerateRandomMatrix(seqNum/2, seqLen)
    seq_pos_matrix_out, InsertMotifType = motifInMatrix(Motiflist, InitMatrix, seqLen)
    seq_positive_name = InsertMotifType.reshape(InsertMotifType.shape[0],)
    seq_pos_label_out = np.ones(seq_pos_matrix_out.shape[0],)

    # GeneRate Negative Dataset
    seq_neg = MatrixToSeq(seq_pos_matrix_out)
    seq_neg_matrix_out = k_mer_shuffle(seq_pos_matrix_out.shape, seq_neg, k=2)
    seq_negative_name = np.zeros([seqNum/2,]) - 1
    seq_neg_label_out = np.zeros(seq_neg_matrix_out.shape[0],)

    # Data Merge
    seq = np.concatenate((seq_pos_matrix_out, seq_neg_matrix_out), axis=0)
    label = np.concatenate((seq_pos_label_out, seq_neg_label_out), axis=0)
    id_tem = np.concatenate((seq_positive_name, seq_negative_name), axis=0)
    index_shuffle = range(seq.shape[0])
    random.shuffle(index_shuffle)
    seq_matrix_out = seq[index_shuffle, :, :]
    label_out = label[index_shuffle]
    id_out = id_tem[index_shuffle].astype("string_")
    outData = [seq_matrix_out, label_out, id_out]
    StoreTrainSet(rootPath=output_dir, allData=outData)




if __name__ == '__main__':
    outputDirAll = "./HDF5/"
    DataDirlist = glob.glob("./motifdata/*")
    seqNum, seqLen = 6000, 1000

    for dataPath in DataDirlist:
        output_dir = outputDirAll + dataPath.split("/")[-1] + "/"
        if os.path.exists(output_dir):
            pass
        elif os.path.isdir(dataPath):
            mkdir(output_dir)
            print("Start" + dataPath)
            GenerateAndSaveData(output_dir, dataPath, seqNum, seqLen)
            print("End" + dataPath)

