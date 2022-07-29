import sys
import pdb
import glob
import re
import h5py
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def calculate(matrixp, matrixq):
    return np.linalg.norm(matrixp - matrixq)


def restoreMTM(FragmentTem):
    """ Calculate MTM from sequences
    Args: FragmentTem

    Return MTM
    """
    MTM = np.zeros((4, 4, FragmentTem.shape[1]))
    MTM[:, 0, 0] = np.count_nonzero(FragmentTem[:, 0, :], axis=0)
    MTM[:, 0, 0] = MTM[:, 0, 0] / MTM[:, 0, 0].sum(axis=0, keepdims=True)
    MTM[:, :, 0] = MTM[:, 0, 0].repeat(4).reshape(4, 4)
    for k in range(1, MTM.shape[2]):
        for pos1 in range(4):
            for pos2 in range(4):
                MTM[pos1, pos2,
                    k] = np.count_nonzero((FragmentTem[:, k - 1, pos1] == 1)
                                          & (FragmentTem[:, k, pos2] == 1))

    MTM_sum = MTM[:, :, 1:].sum(axis=1, keepdims=True)
    MTM[:, :, 1:] = np.divide(MTM[:, :, 1:],
                              MTM_sum,
                              out=np.zeros_like(MTM[:, :, 1:]),
                              where=MTM_sum != 0)
    return MTM


def evaluate(MTMreal, KernelPath):
    """ Calculate Frobenius norm between the ture MTM and predicted MTM
    Args: MTMreal
    KernelPath: The path of output from geneRatemotifs.py

    Return num: the best kernel number
    Frobenius: the smallest norm
    offset: position offset between true MTM and predicted MTM
    """
    FragmentTem = np.load(KernelPath)
    num = re.findall(r"kernel(\d+?).npy", KernelPath)[0]
    if FragmentTem.shape[0] == 0:
        return num, 1000, -1
    MTM = restoreMTM(FragmentTem)

    kernel_len = min(MTM.shape[2], MTMreal.shape[2])
    max_offset = 5
    # if max_offset = 6,
    # I assume offset = 0, 1, 2, 3, 4, 5, 0, -1, -2, -3, -4, -5
    # With offset > 0, restored motif is beyond the real motif;
    # With offset < 0, restored motif is before the real motif.
    norm = np.zeros(2*max_offset)
    for k in range(max_offset):
        # For example, when offset=k=2,
        # real: 0 1 2 3 4 5 6 7 8 9
        # pred: 2 3 4 5 6 7 8 9
        tem = np.zeros(kernel_len - k)
        # Do not consider the first pos
        for pos in range(1, kernel_len - k):
            tem[pos] = calculate(MTMreal[:, :, pos], MTM[:, :, k + pos])
        norm[k] = tem[1:].mean()

        # For example, when offset=k=8, the actual offset=-2
        # real: 2 3 4 5 6 7 8 9
        # pred: 0 1 2 3 4 5 6 7 8 9
        tem = np.zeros(kernel_len - k)
        for pos in range(1, kernel_len - k):
            tem[pos] = calculate(MTMreal[:, :, k + pos], MTM[:, :, pos])
        norm[k + max_offset] = tem[1:].mean()

    Frobenius = norm.min()
    offset = norm.argmin()
    if offset > max_offset:
        offset = max_offset - offset

    return num, Frobenius, offset


def plot(loadpath, savepath, MTMreal):
    """ Plot MTM.
    Args: 
    loadpath: Fragment path or Fragment 
    savepath: save fig

    """
    if type(loadpath) == type(np.array([])):
        FragmentTem = loadpath
    else:
        FragmentTem = np.load(loadpath)
    MTM = restoreMTM(FragmentTem)
    kernel_len = min(MTM.shape[2], MTMreal.shape[2])

    two_decimal = lambda x: format(x, '.2f').rstrip('0').rstrip('.')
    two_decimal = np.vectorize(two_decimal)

    fig, axes = plt.subplots(1, kernel_len, figsize=(2*kernel_len, 2))
    for k in range(kernel_len):
        ax = axes[k]
        if k == 0:
            drawmatrix = MTM[:, 0, k].reshape(4, 1)
            sns.heatmap(drawmatrix,
                        ax=ax,
                        annot=two_decimal(drawmatrix),
                        fmt="s",
                        vmin=0,
                        vmax=1,
                        cmap=sns.light_palette("#2ecc71", as_cmap=True),
                        cbar=False,
                        xticklabels=[" "],
                        yticklabels=["A","C","G","T"],
                        linewidths=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_title("S1", fontsize=15)
        else:
            drawmatrix = MTM[:, :, k].reshape(4, 4)
            sns.heatmap(drawmatrix,
                        ax=ax,
                        annot=two_decimal(drawmatrix),
                        fmt="s",
                        vmin=0,
                        vmax=1,
                        cmap=sns.light_palette("#2ecc71", as_cmap=True),
                        cbar=False,
                        xticklabels=["A","C","G","T"],
                        yticklabels=False,
                        linewidths=0)
            ax.set_title(f"S{k} → S{k+1}", fontsize=15)
        ax.xaxis.tick_top()
        ax.tick_params(bottom=False,top=False,left=False,right=False)

    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def main():
    """ Find and plot the kernel that matches best.
    Real motifs in Motif Path.
    
    """
    MotifPath = "../../result/simulation/Motifs/"
    DatasetPath = "../../external/simulation/"
    Datanamelist = [1, 291, 282, 273]
    name2id = {1: 1, 291: 2, 282: 3, 273: 4}
    DataFrobenius = {}
    resultFile = open(MotifPath+"kernel_offset.txt", "w")
    for name in Datanamelist:
        ResultPath = MotifPath + str(name) + "/MarkonvV/"
        f = h5py.File(
            DatasetPath + str(name) + "/simulationMTM10_20_" + str(name) + ".hdf5",
            "r")
        for key in f.keys():
            MTMreal = f[key]

        AllKernelPath = glob.glob(ResultPath + 'kernel*.npy')
        sel_kernel_num = len(AllKernelPath)
        num = np.zeros(sel_kernel_num,dtype=int)
        Frobenius = np.zeros(sel_kernel_num)
        offset = np.zeros(sel_kernel_num,dtype=int)
        
        for i, KernelPath in enumerate(AllKernelPath):
            num[i], Frobenius[i], offset[i] = evaluate(MTMreal, KernelPath)

        minIndex = Frobenius.argmin()

        plot(ResultPath + 'kernel' + str(num[minIndex]) + '.npy', f'{MotifPath}{name}_kernel_{num[minIndex]}.png', MTMreal)

        print(name, num[minIndex], offset[minIndex], file=resultFile)
        DataFrobenius[name2id[name]] = Frobenius
    resultFile.close()
    DataFrobenius = pd.DataFrame(DataFrobenius)
    DataFrobenius.to_csv(MotifPath + "frobenius.csv", index=None)
    sns.boxplot(data=DataFrobenius)
    plt.ylabel("Frobenius norm")
    plt.xlabel("Complexity")
    plt.savefig(MotifPath + "frobenius.png")
    plt.close()


if __name__ == "__main__":
    main()

