# -*- coding: utf-8 -*-
import pdb
import os
import glob
from multiprocessing import Pool
import sys
import glob
import time
import subprocess, re

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

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]

    return best_gpu


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False


def getpair(KernelLen, KernelNum, RandomSeed, DataName,modeltype):
    # selectGPU = "CUDA_VISIBLE_DEVICES=1 "
    # cmd = selectGPU+ "python torch_main.py"
    cmd = "python torch_main.py"

    Modeltype = modeltype

    KernelNum1 = KernelNum
    tmp_cmd = str(cmd + " " + DataName + " " + KernelLen + " " + str(KernelNum1) + " "
                  + RandomSeed + " " + Modeltype + " " )
    Outpath = DataName.replace("external", "result")

    modelsave_output_filename = Outpath+"/" + Modeltype + "/model_KernelNum-" + str(
        KernelNum) + "kernel_size-" + str(KernelLen) + "_seed-" + str(RandomSeed) + "_batch_size-" + str(
        64) + ".pt"
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
    if os.path.exists(test_prediction_output):
        print("Trained")
        return True
    else:
        return False



def run_model(KernelLen, KernelNum, RandomSeed, DataName,modeltype):
    # selectGPU = "CUDA_VISIBLE_DEVICES=1 "
    # cmd = selectGPU+ "python torch_main.py"
    cmd = "python torch_main.py"

    Modeltype = modeltype

    KernelNum1 = KernelNum
    tmp_cmd = str(cmd + " " + DataName + " " + KernelLen + " " + str(KernelNum1) + " "
                  + RandomSeed + " " + Modeltype + " " )
    Outpath = DataName.replace("external", "result")

    modelsave_output_filename = Outpath+"/" + Modeltype + "/model_KernelNum-" + str(
        KernelNum) + "kernel_size-" + str(KernelLen) + "_seed-" + str(RandomSeed) + "_batch_size-" + str(
        64) + ".pt"
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
    if os.path.exists(test_prediction_output):
        print("Trained")
        return

    print(tmp_cmd)
    os.system(tmp_cmd)


def GetDatalist(rootPath):
    """

    Args:
        rootPath:

    Returns:

    """
    datalist = glob.glob(rootPath+"/*hg19")
    return datalist

def cmdpair():
    usefulpair = []
    ker_size_list = [16, 20]
    randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]
    # randomSeedslist = [0]
    datalist = GetDatalist("../../external/HOCNNLB/Hdf5/")
    modeltypelist = ["MarkonvV"]
    number_of_ker_list = {"Markonv": 128, "MarkonvV": 128, "HOCNNLB": 64}

    for RandomSeed in randomSeedslist:
        for modeltype in modeltypelist:
            for KernelLen in ker_size_list:
                for DataName in datalist:
                    KernelNum = number_of_ker_list[modeltype]
                    trianed = getpair(str(KernelLen), str(KernelNum), str(RandomSeed), DataName, modeltype)
                    if not trianed:
                        usefulpair.append([str(KernelLen), str(KernelNum), str(RandomSeed), DataName, modeltype])
    return usefulpair

if __name__ == '__main__':


    # grid search
    pairlist = cmdpair()
    if len(pairlist)>0:
        proces=10
        for i in range(int(len(pairlist)/proces)+1):
            pool = Pool(processes=min(len(pairlist)- i *proces, proces))
            for j in range(proces*i, min(len(pairlist), proces*(i+1))):
                pair = pairlist[j]
                (KernelLen, KernelNum, RandomSeed, DataName, modeltype) = pair
                pool.apply_async(run_model, (str(KernelLen), str(KernelNum), str(RandomSeed), DataName, modeltype))
                # run_model(str(KernelLen), str(KernelNum), str(RandomSeed), DataName, modeltype)
                time.sleep(3)

            pool.close()
            pool.join()


