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



def run_model(KernelLen, KernelNum, RandomSeed, dataname, Modeltype):
	# selectGPU = "CUDA_VISIBLE_DEVICES=1 "
	# cmd = selectGPU+ "python torch_main.py"
	cmd = "python -u torch_main.py"

	DataName = "../../external/simulation2merShuffle/" + dataname + "/"
	if Modeltype == "MarkonvV":
		KernelLen = str(int(1.5*int(KernelLen)))

	KernelNum1 = KernelNum
	KernelNum2 = 0
	tmp_cmd = str(cmd + " " + DataName + " " + KernelLen + " " + str(KernelNum1) + " "
				  + RandomSeed + " " + Modeltype + " " + str(KernelNum2))

	modelsave_output_filename = "../../result/simulation2merShuffle/"+dataname+"/"+Modeltype+"/model_KernelNum-" + str(
		KernelNum) + "kernel_size-" + str(KernelLen) + "_seed-" + str(RandomSeed) + "_batch_size-" + str(
		64) + ".pkl"

	test_prediction_output = modelsave_output_filename.replace("/model_KernelNum-", "/Report_KernelNum-")
	
	mkdir("../../result/simulation2merShuffle/"+dataname+"/"+Modeltype+"/")

	# retrain (only used for DEBUG)
	retrain = False
	if retrain:
		os.remove(test_prediction_output)

	if os.path.exists(test_prediction_output):
		print("already Trained")
		# print(test_prediction_output)
		return 0, 0
	# else:
	# 	tmp_cmd += " > "+modelsave_output_filename.replace(".pkl",".out") + " 2>&1 "
	print(tmp_cmd)


	os.system(tmp_cmd)


if __name__ == '__main__':
	# grid search
	ker_size_list = [10]
	number_of_ker_list = [128]
	randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927, 42, 48, 100, 420, 60, 320, 7767, 51]
	datasetnamelist = [1,291,282,273]
	pool = Pool(processes=16)#len(datasetnamelist)*len(ker_size_list)*len(number_of_ker_list)*len(randomSeedslist))
	for dataname in datasetnamelist:
		for RandomSeed in randomSeedslist:
			for KernelNum in number_of_ker_list:
				for KernelLen in ker_size_list:
					pool.apply_async(run_model, (str(KernelLen), str(KernelNum), str(RandomSeed), str(dataname), "MarkonvV"))
					time.sleep(5)
					pool.apply_async(run_model, (str(KernelLen), str(KernelNum*4), str(RandomSeed), str(dataname), "CNN"))
					time.sleep(5)
					pool.apply_async(run_model, (str(KernelLen), str(KernelNum), str(RandomSeed), str(dataname), "DANQ"))
					time.sleep(5)
					pool.apply_async(run_model, (str(KernelLen), str(KernelNum), str(RandomSeed), str(dataname), "LSTM"))
					time.sleep(5)
					pool.apply_async(run_model, (str(KernelLen), str(KernelNum), str(RandomSeed), str(dataname), "DANQS"))
					time.sleep(5)
					pool.apply_async(run_model, (str(KernelLen), str(KernelNum), str(RandomSeed), str(dataname), "LSTM_last"))
					time.sleep(5)
	pool.close()
	pool.join()
