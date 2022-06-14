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



def run_model(KernelLen, KernelNum, RandomSeed, dataname,Modeltype):
    # selectGPU = "CUDA_VISIBLE_DEVICES=1 "
    # cmd = selectGPU+ "python torch_main.py"
    cmd = "python torch_main.py"

    DataName = "../../external/simulation/" + dataname + "/"

    KernelNum1 = KernelNum
    KernelNum2 = 0
    tmp_cmd = str(cmd + " " + DataName + " " + KernelLen + " " + str(KernelNum1) + " "
                  + RandomSeed + " " + Modeltype + " " + str(KernelNum2))

    print(tmp_cmd)
    os.system(tmp_cmd)


def GetDatalist(rootPath):
    """

    Args:
        rootPath:

    Returns:

    """
    datalist = glob.glob(rootPath+"/*")
    datanamelist = []

    for datapath in datalist:
        datanamelist.append(datapath.split("/")[-1])

    return datanamelist


if __name__ == '__main__':
    # grid search
    KernelLen = 10
    KernelNum = 128
    RandomSeed = 0
    datasetnamelist = [1, 291, 282, 273]
    # modeltypelist = ["Markonv","MarkonvR","MarkonvV","CNN"]
    modeltypelist = ["Markonv","CNN"]
    pool = Pool(processes=len(datasetnamelist))
    for dataname in datasetnamelist:
        for modeltype in modeltypelist:
            # run_model(str(KernelLen), str(KernelNum), str(RandomSeed), str(dataname), modeltype)
            pool.apply_async(run_model, (str(KernelLen), str(KernelNum), str(RandomSeed), str(dataname),modeltype))
            time.sleep(10)
    pool.close()
    pool.join()
