import os
import sys
import h5py
import subprocess, re
import numpy as np
import pdb
import copy
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_auc_score


def loadData(path, Modeltype="Markonv"):
    """
    load data

    :param path:
    :return:
    """
    f_train = h5py.File(path + "/train.hdf5", "r")
    TrainX = f_train["sequences"][()]
    TrainY = f_train["labs"][()]
    f_train.close()

    f_test = h5py.File(path + "/test.hdf5", "r")
    TestX = f_test["sequences"][()]
    TestY = f_test["labs"][()]
    f_test.close()

    if Modeltype == "BConv":
        dim = TrainX.shape[1] * TrainX.shape[2]
        return [[TrainX.reshape(TrainX.shape[0], dim), TrainY],
                [TestX.reshape(TestX.shape[0], dim), TestY]]
    else:
        return [[TrainX, TrainY], [TestX, TestY]]

def readmodelpath(file):
    """

    Args:
        file:

    Returns:

    """
    path = open(file,"r").readlines()[0]
    kernel_size= int(path.split("/")[-1].split("-")[2].split("_")[0])
    number_of_kernel= int(path.split("/")[-1].split("-")[1].split("kernel")[0])
    return path.replace(".hdf5",".checkpointer.pt"),kernel_size, number_of_kernel


def kernelSelect(model, testX, TestY):
    """
    Only use model.markonv.Kernel_Full_4DTensor
    Args:
        model:
        testX:
        TestY:

    Returns:

    """
    y_pred = model(testX)
    kernelweight = model.markonv.Kernel_Full_4DTensor.data
    test_auc = roc_auc_score(TestY,y_pred.cpu().detach())

    KernelAuclist = []


    for i in range(kernelweight.shape[3]):
        # mask a kernel in temkernelweights
        temkernelweights = copy.deepcopy(kernelweight)
        tem = torch.zeros_like(temkernelweights[:,:,:,i])
        temkernelweights[:,:,:,i] = tem

        # use the masked model to predict
        model.markonv.Kernel_Full_4DTensor.data = temkernelweights
        tempre = model(testX)
        kernelauctem = roc_auc_score(TestY, tempre.cpu().detach())
        KernelAuclist.append(test_auc- kernelauctem)

    kernelindexTem = np.asarray(KernelAuclist)

    RankIndex = np.argsort(kernelindexTem)[::-1]

    return RankIndex[:10]

def draw(kernels, savepath,RankIndex):
    """

    Args:
        kernels:
        savepath:

    Returns:

    """
    kernel_len = kernels.shape[1]

    two_decimal = lambda x: format(x, '.2f').rstrip('0').rstrip('.')
    two_decimal = np.vectorize(two_decimal)

    for i in range(kernels.shape[0]):
        savepathtmp = savepath + "motifs_kernel_"+str(RankIndex[i])+".png"
        kernel = kernels[i]
        kernel = kernel

        fig, axes = plt.subplots(1, kernel_len, figsize=(kernel_len*2, 2))
        for k in range(kernel_len):
            ax = axes[k]
            drawmatrix = kernel[k, :, :]
            sns.heatmap(drawmatrix,
                        ax=ax,
                        annot=two_decimal(drawmatrix),
                        fmt="s",
                        vmin=-1,
                        vmax=1,
                        # cmap=sns.light_palette("#2ecc71", as_cmap=True),
                        cmap="Greys",
                        cbar=False,
                        xticklabels=["A","C","G","T"],
                        yticklabels=False,
                        linewidths=0)
            # ax.set_title(f"S{k} → S{k+1}", fontsize=15)
            ax.xaxis.tick_top()
            ax.tick_params(bottom=False,top=False,left=False,right=False)

        plt.savefig(savepathtmp, bbox_inches='tight')
        plt.close()


def GenerateFragments(X,onehot,model,kernelSize,path,RankIndex=None):
    """
    scaner the sequences and find the highest Score fragments
    Args:
        X:
        onehot:
        Scanner:

    Returns:

    """

    scores = model.markonv(X).permute((0,2,1)).cpu().detach().numpy()
    for i in range(scores.shape[2]):
        if (RankIndex is None) or (i in RankIndex):
            ### kernel
            position = np.argmax(scores[:,:,i],axis=1)
            threshold = np.mean(np.max(scores[:,:,i],axis=1))
            FragmentTem = []
            for j in range(scores.shape[0]):
                # batch
                if scores[j,position[j],i]>threshold and position[j] + int(model.markonv.k_weights[1, 0, i]) < onehot.shape[1]:
                    FragmentTem.append(
                        onehot[j, position[j] + int(model.markonv.k_weights[0, 0, i]):position[j] + int(model.markonv.k_weights[1, 0, i]), :])
            ### store the fragments for all kernels
            Fragments = np.asarray(FragmentTem)
            np.save(path+"/kernel"+str(i),Fragments)
            plt.hist(np.max(scores[:,:,i],axis=1))

            plt.savefig(path+"/kernel_"+str(i)+".png")
            plt.close()


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

def restoreMotif(net, RankIndex):


    kernels = net.markonv.Kernel_Full_4DTensor.data
    kernels = kernels.cpu().numpy()

    outputkernel = []

    for i in range(len(RankIndex)):
        outputkernel.append(kernels[:,:,:,RankIndex[i]])

    return np.asarray(outputkernel)



def main():
    """

    Returns:

    """
    DataPath = "../../external/simulation2merShuffle/"
    resultpath = "../../result/simulation2merShuffle/"
    OutputPath = "../../result/simulation2merShuffle/Motifs"
    mkdir(OutputPath)
    Datanamelist = [1,291,282,273]
    model_type = "MarkonvV"

    for name in Datanamelist:
        path = DataPath+ str(name)+"/"

        ### load dataset
        data_set = loadData(path)
        training_set, test_set = data_set
        X_test_onehot, Y_test = test_set
        X_test = torch.from_numpy(X_test_onehot)
        Y_test = torch.from_numpy(Y_test)
        X_test = X_test.to(device)

        ### load model
        modelpath, kernel_size, number_of_kernel = readmodelpath(resultpath+str(name)+"/"+model_type+"/bestmodel.txt")
        net = MarkonvModel(kernel_size,number_of_kernel,1,model_type)
        net.load_state_dict(torch.load(modelpath, map_location=device))
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            ##### select useful kernels
            RankIndex = kernelSelect(net, X_test, Y_test)

            ### kernels scan sequences and find the most similar fragment
            path = OutputPath+"/"+str(name)+"/"+model_type+"/"
            mkdir(path)

            kernels = restoreMotif(net, RankIndex)

            draw(kernels, path, RankIndex)

            GenerateFragments(X_test, X_test_onehot,net,kernel_size,path,RankIndex)



if __name__ == '__main__':

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
            assert m, "Couldnt parse " + line
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
            if gpu_id not in [0,3,4]:
                memory_gpu_map.append((memory,gpu_id))
        best_memory, best_gpu = sorted(memory_gpu_map)[0]

        return best_gpu

    GPUID = '3'#str(pick_gpu_lowest_memory())
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

    import torch
    from torch_build import MarkonvModel, TorchDataset

    #torch.cuda.set_device(int(GPUID))
    main()
