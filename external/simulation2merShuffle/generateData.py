import h5py
import numpy as np
import glob
import os
from ushuffle import Shuffler, shuffle
import random
import pdb

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



def sperateData(X, Y):
    """

    """
    Pos_Pos = np.where(Y>0)[0]
    Pos_Neg = np.where(Y==0)[0]

    Positive = X[Pos_Pos,:,:]
    Negative = X[Pos_Neg,:,:]

    return Positive, Negative


def loaddata(path):
    """

    """
    f_train = h5py.File(path + "/train.hdf5", "r")
    TrainX = f_train["sequences"][()]
    TrainY = f_train["labs"][()]
    f_train.close()

    f_test = h5py.File(path + "/test.hdf5", "r")
    TestX = f_test["sequences"][()]
    TestY = f_test["labs"][()]
    f_test.close()

    TrainPositive,TrainNegative = sperateData(TrainX, TrainY)
    TestPositive,TestNegative = sperateData(TestX, TestY)


    return TrainPositive,TrainNegative, TestPositive,TestNegative


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




def StoreTrainSet(rootPath, allData,name):
    """
    store different dataset
    :param rootPath:
    :param allData: All data
    """
    dataNum = allData[1].shape[0]


    mkdir(rootPath)
    training_path = rootPath+ "/train.hdf5"
    test_path = rootPath+"/test.hdf5"

    f_train = h5py.File(training_path, "w")
    f_test = h5py.File(test_path, "w")

    f_train.create_dataset("sequences",data = allData[0])
    f_train.create_dataset("labs",data=allData[1])
    f_train.close()
    f_test.create_dataset("sequences",data=allData[2])
    f_test.create_dataset("labs",data=allData[3])
    f_test.close()

def ConcentrateData(TrainPositive, TrainNegative, TestPositive, TestNegative):

    labelTrainPos = np.ones((TrainPositive.shape[0],))
    labelTrainNeg = np.zeros((TrainNegative.shape[0],))

    labelTestPos = np.ones((TestPositive.shape[0],))
    labelTestNeg = np.zeros((TestNegative.shape[0],))
    TrainX = np.concatenate([TrainPositive, TrainNegative],axis=0)
    TrainY = np.concatenate([labelTrainPos, labelTrainNeg],axis=0)

    TestX = np.concatenate([TestPositive, TestNegative],axis=0)
    TestY = np.concatenate([labelTestPos, labelTestNeg],axis=0)


    number = TrainPositive.shape[0] + TrainNegative.shape[0]
    index_shuffle = list(range(number))
    random.shuffle(index_shuffle)

    TrainX = TrainX[index_shuffle, :, :]
    TrainY = TrainY[index_shuffle]


    number = TestPositive.shape[0] + TestNegative.shape[0]
    index_shuffle = list(range(number))
    random.shuffle(index_shuffle)
    TestX = TestX[index_shuffle, :, :]
    TestY = TestY[index_shuffle]

    outData = [TrainX, TrainY,TestX, TestY]
    return outData

def main():
    """

    """

    namelist = [1,291,282,273]

    for name in namelist:
        name = str(name)
        rootPath = "./"+name+"/"
        path = "../simulation/"+name+"/"
        ## load original data

        TrainPositive, TrainNegative, TestPositive, TestNegative = loaddata(path)
        TrainNegativeseq = MatrixToSeq(TrainPositive)
        TestNegativeseq = MatrixToSeq(TestPositive)

        ## shuffle positive data to replace the old negative data
        TrainNegative = k_mer_shuffle(TrainPositive.shape, TrainNegativeseq, k=2)
        TestNegative = k_mer_shuffle(TestPositive.shape, TestNegativeseq, k=2)


        ## save data
        allData = ConcentrateData(TrainPositive, TrainNegative, TestPositive, TestNegative)
        StoreTrainSet(rootPath, allData, name)







if __name__ == '__main__':
    main()