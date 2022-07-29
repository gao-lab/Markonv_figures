# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import pandas as pd
import seaborn as sns
import pdb
plt.switch_backend('agg')
import copy
def loadResult(path,sel_kernum=None):
	"""

	:param path:
	:param modelType:
	:return: auc list
	"""

	filelist = glob.glob(path+"/Report*")
	auclist = []
	bestAUC = 0
	for file in filelist:
		kernel_num = int(file.split('_')[-5][10:13])
		seed = int(file.split('_')[-3][5:])

		with open(file, "rb") as f:
			ret = (pickle.load(f)).tolist()
			if ret["test_auc"] > bestAUC:
				bestAUC = ret["test_auc"]
				tmp_path = file.replace("pkl", "hdf5")
				BestModel = tmp_path.replace("/Report_KernelNum-", "/model_KernelNum-")
			auc = ret["test_auc"]
		auclist.append(auc)
	f = open(path + "/bestmodel.txt", mode="w")
	f.write(BestModel)
	return auclist

def Draw(path, ModeltypeDict, dataname):
	"""

	:param path:
	:param ModeltypeDict:
	:return:
	"""
	Pddict = pd.DataFrame(ModeltypeDict)
	#print('std')
	#print(Pddict.std())
	sns.boxplot(data=Pddict)
	plt.ylim(0.5, 1)
	plt.ylabel("AUROC")
	plt.xlabel(str(dataname))
	# Pddict.boxplot()
	plt.savefig(path + "auc.png")
	plt.close('all')



def Main():
	"""

	:return:
	"""

	path = "../../result/JasparSimu/HDF5//"
	outputpath = "../../result/JasparSimu//picture/"
	mkdir(outputpath)
	datasetnamelist = [1,2,3,4]

	for dataname in datasetnamelist:
		resultPath = path+str(dataname)+"/"
		ModelType = ["CNN","MarkonvV"]

		ResultDict = {}
		for Mtype in ModelType:
			resultPath = path+str(dataname)+"/"
			auclist = loadResult(resultPath+Mtype+"/")
			if Mtype == "MarkonvV":
				ResultDict["Markonv-based network"] = auclist
			elif Mtype == "CNN":
				ResultDict["Convolution-based network"] = auclist
			elif Mtype == "DANQ":
				ResultDict["CNN-RNN network"] = auclist
			mkdir("../../result/JasparSimu/files.2/")

			np.savetxt("../../result/JasparSimu/files.2/"+str(dataname)+"_"+Mtype+"_auc.txt", np.asarray(auclist))

		Draw(outputpath+str(dataname), ResultDict, dataname)




'''def GetBestmodel(SimulationResultRoot):
	"""

	:return:
	"""
	model = ["BCNN","CNN"]
	lossDict = {}
	for mode in model:
		path = glob.glob(SimulationResultRoot+"/"+mode+"/Report*")
		BestModellosslist=[]
		bestAUC= 0
		for file in path:

			with open(file, "r") as f:
				tmp_dir = (pickle.load(f)).tolist()
				try:
					if tmp_dir["test_auc"] > bestAUC:
						bestAUC = tmp_dir["test_auc"]
						BestModellosslist = tmp_dir["loss"]
				except:
					import pdb
					pdb.set_trace()
		lossDict[mode] = BestModellosslist
	return lossDict
def flat_record(rec):
	try:
		output = np.array([x for y in rec for x in y])
	except:
		output = np.array(rec)
	return output'''


def mkdir(path):
	"""
	Create a directory
	:param path: Directory path
	:return:
	"""
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False

'''def draw_history(lossDict,path):


	save_root = path+"/history/"
	mkdir(save_root)
	color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

	mode_lst = ["BCNN","CNN"]

	plt.clf()

	plt.title("loss history:  simulation dataset")

	plt.xlabel("epoch")

	plt.ylabel("simulation dataset")

	for idx,mode in enumerate(mode_lst):
		label = mode + "-based model"
		tmp_data = lossDict[mode]
		y = [x for it in tmp_data for x in it]
		plt.plot(np.arange(len(y)),np.array(y),label=label,color=color_list[idx]) #,label=mode,color=color_list[idx]
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
		  fancybox=True, shadow=True, ncol=5)
	plt.savefig(save_root+"loss.eps", format="eps")
	plt.savefig(save_root+"loss.png")



def DrawHistory():
	"""

	:return:
	"""
	SimulationResultRoot = "/rd2/lijy/BCNN/result/simulation2/1MTMTraining/"
	GetBestmodel(SimulationResultRoot)
	result = GetBestmodel(SimulationResultRoot)
	draw_history(result, SimulationResultRoot)'''




if __name__ == '__main__':
	Main()
	# DrawHistory()

