# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import pandas as pd
import seaborn as sns
import pdb
# plt.switch_backend('agg')
import plotly.express as px







def main():
    """

    """
    ####load feature

    modelsize = {"Markonv-basde bonito":9115728, "Bonito":27008104}

    result = pd.read_csv("../../result/bonito/bonito_multiple_seeds.tsv",sep="\t")

    # read_acc = {"Markonv-basde bonito":result["read_markonv"], "Bonito":result["read_conv"]}

    namelist = ["Markonv-basde bonito"]*len(result["read_markonv"])+["Bonito"]*len(result["read_markonv"])
    Readacc = list(result["read_markonv"])+list(result["read_conv"])
    sizelist = [(9.115728/3)**2]*len(result["read_markonv"])+[(27.008104/3)**2]*len(result["read_markonv"])
    consensusacc =list(result["assembly_markonv"])+list(result["assembly_conv"])
    colorlist = ["read"]*len(result["read_markonv"])+["Bonito"]*len(result["read_markonv"])
    # fig = plt.figure(figsize=(8, 14))


    df = pd.DataFrame({"read accuracy":Readacc,"model size":sizelist,"consensus accuracy":consensusacc,"model name":namelist})
    fig = px.scatter(data_frame=df, x="consensus accuracy", y="read accuracy", size="model size", color="model name")
    # ax.set_xlabel("consensus accuracy")
    # ax.set_ylabel("read accuracy")
    # ax.set(title='model ')
    fig.update_layout({"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)"})
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.show()








if __name__ == '__main__':
    main()

