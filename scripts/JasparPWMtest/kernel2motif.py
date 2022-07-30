import sys
import pdb
import glob
import re
import h5py
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
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


def plot(loadpath, savepath):
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
    # kernel_len = min(MTM.shape[2], MTMreal.shape[2])
    kernel_len = MTM.shape[2]

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



def plotPWM(loadpath):
    """ Plot MTM.
    Args:
    loadpath: Fragment path or Fragment

    """
    if type(loadpath) == type(np.array([])):
        FragmentTem = loadpath
    else:
        FragmentTem = np.load(loadpath)
    pwm = FragmentTem.sum(axis=0)
    pwm = pwm/pwm.sum(axis=1,keepdims=True)
    savepath = "../../result/JasparSimu/PWMMotifs2/"+loadpath.split("/")[-1].replace(".npy",".png")
    mkdir("../../result/JasparSimu/PWMMotifs2/")
    drawseqlogo(pwm, savepath)


def ChangePwmtoInputFormat(pwm):
    """

    """
    output = []
    sortlist = ["A","C","G","T"]

    for i in range(pwm.shape[0]):
        output.append([])

        ShanoyE = 0
        for m in range(4):
            if pwm[i,m]>0:
                ShanoyE = ShanoyE - pwm[i,m]*np.log(pwm[i,m]) / np.log(2)

        IC = np.log(4)/np.log(2) - (ShanoyE)
        for j in range(4):
            output[i].append([sortlist[j], pwm[i,j]*IC])

    return output


def letterAt(letter, x, y, yscale=1, ax=None):
    fp = FontProperties(family="Arial", weight="bold")
    globscale = 1.35
    LETTERS = {"T": TextPath((-0.305, 0), "T", size=1, prop=fp),
               "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
               "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
               "C": TextPath((-0.366, 0), "C", size=1, prop=fp)}
    COLOR_SCHEME = {'G': 'orange',
                    'A': 'red',
                    'C': 'blue',
                    'T': 'darkgreen'}

    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p



def drawseqlogo(pwm, path):
    """

    """

    fig, ax = plt.subplots(figsize=(10,3))

    all_scores = ChangePwmtoInputFormat(pwm)
    x = 1
    maxi = 0
    for scores in all_scores:
        y = 0
        for base, score in scores:
            letterAt(base, x,y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)
    plt.rcParams.update({'font.size': 20})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(range(1,x))
    plt.xlim((0, x))
    # plt.ylim((0, maxi))
    plt.ylim((0, 2))
    plt.xlabel("Position",fontsize=25)
    plt.ylabel("Information \n content",fontsize=25)
    plt.tight_layout()
    plt.savefig(path)

def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    import os
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)




def main():
    """ Find and plot the kernel that matches best.
    Real motifs in Motif Path.
    
    """
    MotifPath = "../../result/JasparSimu/Motifs2/"
    outputpath = "../../result/JasparSimu/Motifs2/figure/"
    DatasetPath = "../../external/JasparSimu/HDF5/"
    Datanamelist = [1, 291, 282, 273]
    name2id = {1: 1, 291: 2, 282: 3, 273: 4}
    DataFrobenius = {}
    mkdir(outputpath)
    for name in Datanamelist:
        ResultPath = MotifPath + str(name) + "/MarkonvV/"
        # f = h5py.File(
        #     DatasetPath + str(name) + "/simulationMTM10_20_" + str(name) + ".hdf5",
        #     "r")
        # for key in f.keys():
        #     MTMreal = f[key]

        AllKernelPath = glob.glob(ResultPath + 'kernel*.npy')
        sel_kernel_num = len(AllKernelPath)
        # num = np.arange(sel_kernel_num,dtype=int)
        # Frobenius = np.zeros(sel_kernel_num)
        # offset = np.zeros(sel_kernel_num,dtype=int)
        
        # for i, KernelPath in enumerate(AllKernelPath):
        #     num[i], Frobenius[i], offset[i] = evaluate(MTMreal, KernelPath)

        # minIndex = Frobenius.argmin()
        for i in range(sel_kernel_num):
            num = int(AllKernelPath[i].split("/")[-1].split(".")[0].split("kernel")[-1])
            plot(AllKernelPath[i], f'{outputpath}{name}_kernel_{num}.png')

            plotPWM(AllKernelPath[i])




if __name__ == "__main__":
    main()

