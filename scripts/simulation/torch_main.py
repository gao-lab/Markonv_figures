import os
import pdb

import torch
import sys
import h5py
import subprocess, re
from torch_build import *

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""
    memory_gpu_map = []
    for (gpu_id, memory) in gpu_memory_map().items():
        #if gpu_id not in [0,3,4]:
        memory_gpu_map.append((memory,gpu_id))

    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def GridSearch(data_set, KernelLen, KernelNum, RandomSeed, Modeltype, path,GPUID="0",
               KernelNum2=0,batch_size=32,epoch_scheme=1000):
    """
    grid search hyper- parameters
    :param data_set:
    :param KernelLen:
    :param KernelNum:
    :param RandomSeed:
    :param Modeltype: "CNN", "MarkonvV"
    :param path:
    :param GPUID:
    :param batch_size:
    :param epoch_scheme:
    :return:
    """
    input_shape = data_set[0][0].shape[1:]
    
    #if Modeltype=="CNN":
    #    trainCNN(path, data_set, KernelNum, KernelLen,
    #                 RandomSeed, batch_size, epoch_scheme,GPUID=GPUID,outputName="CNN")
    #elif Modeltype in ["Markonv","MarkonvR","MarkonvV"]:
    #    trainMarkonv(path, data_set, KernelNum, KernelLen,
    #                 RandomSeed, batch_size, epoch_scheme,GPUID=GPUID,outputName=Modeltype)
    #else:
    #    raise ValueError('Invalid type: %s' % type)

    trainMarkonv(path, data_set, KernelNum, KernelLen,
                     RandomSeed, batch_size, epoch_scheme,GPUID=GPUID,outputName=Modeltype)


def loadData(path, Modeltype):
    """
    load data
    
    :param path:
    :return:
    """
    f_train = h5py.File(path + "/train.hdf5","r")
    TrainX = f_train["sequences"][()]
    TrainY = f_train["labs"][()]
    f_train.close()
    
    f_test = h5py.File(path + "/test.hdf5","r")
    TestX = f_test["sequences"][()]
    TestY = f_test["labs"][()]
    f_test.close()
    return [[TrainX, TrainY], [TestX, TestY]]

    
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
    torch.set_num_threads(3)
    GPUID = str(pick_gpu_lowest_memory())
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
    torch.cuda.set_device(int(GPUID))
    KernelNum2 = 0

    DataPath = sys.argv[1]
    KernelLen = int(sys.argv[2])
    KernelNum = int(sys.argv[3])
    RandomSeed = int(sys.argv[4])
    Modeltype = sys.argv[5]
    if len(sys.argv)>6:
        KernelNum2 = int(sys.argv[6])
    else:
        KernelNum2 = 0
    # os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    data_set = loadData(DataPath, Modeltype)
    
    Outpath = DataPath.replace("../../external/", "../../result/")
    mkdir(Outpath)

    GridSearch(data_set, KernelLen, KernelNum, RandomSeed,
               Modeltype, Outpath,GPUID=GPUID,KernelNum2=KernelNum2, batch_size=64, epoch_scheme=1000)