import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
import pdb

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


def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

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


def drawseqlogo(path):
    """

    """
    pwm = np.loadtxt(path)
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
    plt.savefig(path.replace(".txt",".SeqLogo.png"))


def main():
    import glob
    pathlist = glob.glob("./*.txt")

    for path in pathlist:
        drawseqlogo(path)



if __name__ == '__main__':
    main()