import numpy as np
import pdb
import os
import h5py
import random
from sklearn.model_selection import StratifiedKFold
import gzip
import glob


def read_seq(seq_file):
    seq_list = []
    namelist = []
    labellist = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            line = line.decode('UTF-8')
            if line[0] == '>':
                name = line[1:-1]
                namelist.append(name)
                label = int(name.split(":")[-1])
                labellist.append(label)
                if len(seq):
                    seq_list.append(seq)
                seq = ''
            else:
                seq = seq + line[:-1]


        if len(seq):
            seq_list.append(seq)

    return np.array(namelist), np.array(seq_list), np.asarray(labellist)


def seq_to_matrix(seq, seq_matrix, seq_order):
    '''
    change target 3D tensor according to sequence and order
    :param seq:
    :param seq_matrix:
    :param seq_order:
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


def StoreTrainSet(rootPath, allData):
    """
    store different dataset
    :param rootPath:
    :param allData: All data
    """

    mkdir(rootPath)
    training_path = rootPath + "/train.hdf5"
    test_path = rootPath + "/test.hdf5"

    f_train = h5py.File(training_path, "w")
    f_test = h5py.File(test_path, "w")

    f_train.create_dataset("sequences", data=allData[0])
    f_train.create_dataset("labs", data=allData[1])
    f_train.create_dataset("seq_idx", data=allData[2])
    f_train.close()
    print("Train: ", allData[0].shape)

    f_test.create_dataset("sequences", data=allData[3])
    f_test.create_dataset("labs", data=allData[4])
    f_test.create_dataset("seq_idx", data=np.asarray(allData[5],dtype="|S35"))
    f_test.close()
    print("test: ", allData[3].shape)


def cross_validation(number_of_folds, total_number, random_seeds=233):
    """
    :param number_of_folds:
    :param total_number:
    :param random_seeds:
    :return:
    """
    x = np.zeros((total_number,), dtype=np.int)
    split_iterator = StratifiedKFold(n_splits=number_of_folds, random_state=random_seeds, shuffle=True)
    split_train_index_and_test_index_list = [
        (train_index, test_index)
        for train_index, test_index in split_iterator.split(x, x)
    ]
    return (split_train_index_and_test_index_list)


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
    id_train = data_id[split_index_list[fold][0].tolist()]
    x_train = data_x[split_index_list[fold][0].tolist()]
    y_train = data_y[split_index_list[fold][0].tolist()]
    id_test = data_id[split_index_list[fold][1].tolist()]
    x_test = data_x[split_index_list[fold][1].tolist()]
    y_test = data_y[split_index_list[fold][1].tolist()]
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


def genarate_matrix_for_train(seq_shape, seq_series):
    """
    sequence to one-hot matrix
    :param shape: (seq number, sequence_length, 4)
    :param seq_series:
    :return:seq
    """
    seq_matrix = np.zeros(seq_shape)
    for i in range(seq_series.shape[0]):
        seq_tem = seq_series[i]
        seq_matrix = seq_to_matrix(seq_tem, seq_matrix, i)
    return seq_matrix


def run(SeqTrainPath, SeqTestPath, OutputDir):
    """
    seq to one-hot matrix and save into HDF5
    :return:
    """
    SeqLen = 0

    # training data
    Trainid, TrainMatrix, Trainlabel = read_seq(SeqTrainPath)
    TrainMatrix_shape = TrainMatrix.shape[0]
    for i in range(TrainMatrix_shape):
        SeqLen = max(SeqLen, len(TrainMatrix[i]))
    TrainMatrix = genarate_matrix_for_train((TrainMatrix_shape, SeqLen, 4), TrainMatrix)

    # test data
    TestName, TestSeq, Testlabel = read_seq(SeqTestPath)
    Testshape = TestSeq.shape[0]
    for i in range(Testshape):
        SeqLen = max(SeqLen, len(TestSeq[i]))

    TestMatrix = genarate_matrix_for_train((Testshape, SeqLen, 4), TestSeq)

    # save data

    index_shuffle = list(range(TrainMatrix_shape))
    random.shuffle(index_shuffle)
    TrainMatrix = TrainMatrix[index_shuffle, :, :]
    Trainlabel = Trainlabel[index_shuffle]
    Trainid = Trainid[index_shuffle].astype("string_")
    outData = [TrainMatrix, Trainlabel, Trainid, TestMatrix, Testlabel, TestName]
    StoreTrainSet(rootPath=OutputDir, allData=outData)


def main():
    """
    :return:
    """
    rootPath = "../fasta/"

    Dirlist = glob.glob(rootPath + "/*")
    for Dir in Dirlist:
        Tmplist = glob.glob(Dir + "/*")
        if len(Tmplist) >= 0:
            print(Dir)
            SeqTrainPath = Dir + "/train/1/sequence.fa.gz"
            SeqTestPath = Dir + "/test/1/sequence.fa.gz"
            OutputDir = Dir.replace("fasta", "Hdf5").replace("RBPdata1201","") + "/"
            mkdir(OutputDir)
            run(SeqTrainPath, SeqTestPath, OutputDir)


if __name__ == '__main__':
    main()