# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:23:12 2023

@author: Hongjie Xing

"""

# 本文件用于使用训练好的模型预测流场
import pickle
from utils.train_functions import *
from utils.functions import *
from model.UNetEx import UNetEx
import configparser
import os
import re
import operator
from scipy.ndimage import gaussian_filter
import time
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
config = configparser.ConfigParser()
config.read("./config/config.ini")

# 设置卷积核大小
kernel_size = int(config["net_parameter"]["kernel_size"])
# 设置卷积层channel数目
filters = [int(i) for i in config["net_parameter"]["filters"].split(",")]
# 设置batch_norm和weight_norm
bn = bool(int(config["net_parameter"]["batch_norm"]))
wn = bool(int(config["net_parameter"]["weight_norm"]))
# 构建模型
model = UNetEx(2, 1, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
# 加载模型参数
modelNum = 9861  #9861
modelName = 'DeepCFD_' + str(modelNum) + '.pdparams'
model.set_state_dict(
        paddle.load(os.path.join(config["path"]["save_path"], modelName)))

# 获得文件编号

# directoryPath = "./data/testData/"
# directoryPath = "./paperFig/data/"
directoryPath = "F:/PredictResults/data/testData2/"
directoryPathLPF = directoryPath + "result_LPFField"
directoryPathLDF = directoryPath + "result_LDFField"
directoryPathRad = directoryPath + "result_RadField"

sampleNumMatch = re.compile(r'\d+_\d+')
matrixSize = 192

sampleNumLDF = []
LDFfiles = [f for f in os.listdir(directoryPathLDF) if f.endswith(".txt")]
for file_name in os.listdir(directoryPathLDF) :
    match = sampleNumMatch.search(file_name)
    if match :
        #sampleNumLDF.append(match.group())
        sampleNumLDF.append(file_name[3:-4])

sampleNumLPF = []
LPFfiles = [f for f in os.listdir(directoryPathLPF) if f.endswith(".txt")]
for file_name in os.listdir(directoryPathLPF) :
    match = sampleNumMatch.search(file_name)
    if match :
        #sampleNumLPF.append(match.group())
        sampleNumLPF.append(file_name[3:-4])
        
sampleNumRad = []
Radfiles = [f for f in os.listdir(directoryPathRad) if f.endswith(".txt")]
for file_name in Radfiles :
    match = sampleNumMatch.search(file_name)
    if match :
        #sampleNumRad.append(match.group())
        sampleNumRad.append(file_name[3:-4])

assert (operator.eq(sampleNumLDF, sampleNumLPF))    
assert (operator.eq(sampleNumLDF, sampleNumRad))
totalPredictNum = len(sampleNumRad)
print("Using model: ", modelName)
print("total number of sample to be predicted = ", totalPredictNum)

# 加载数据集并处理
matricesLDFLPF = [] #先LDF，后LPF，顺序不能乱
matricesRad = []

random.seed(42)
file_indices = list(range(totalPredictNum))
selectPredictNum = 799 #763
print('selectPredictNum = ', selectPredictNum)
random_indices = random.sample(file_indices, selectPredictNum)

#for i in range (0, totalPredictNum) :
for i in random_indices:
    #print(i)
    sFileLPF = "LPF{}.txt".format(sampleNumLDF[i])
    sFileLDF = "LDF{}.txt".format(sampleNumLDF[i])
    sFileRad = "Rad{}.txt".format(sampleNumRad[i])
    if (( (sFileLPF in LPFfiles) and (sFileLDF in LDFfiles) ) and (sFileRad in Radfiles)) :
        matrixLDF = []
        matrixLPF = []
        matrixLDFLPF = []
        with open(os.path.join(directoryPathLDF, sFileLDF), 'r') as rLDF:
            matrixLDF = np.loadtxt(rLDF)
            #需要将LDF进行上下对称操作
            matrixLDF = matrixLDF[::-1]
        with open(os.path.join(directoryPathLPF, sFileLPF), 'r') as rLPF:
            matrixLPF = np.loadtxt(rLPF)
            #需要将LDF进行上下对称操作
            matrixLPF = matrixLPF[::-1]
            # 使用LPF的模式，倒数用下面一行
            # np.seterr(divide="ignore")
            # matrixLPF = np.where(matrixLPF != 0, 150.0 / matrixLPF / matrixLPF, 0)
        matrixLDFLPF = np.stack((matrixLDF, matrixLPF), axis=0) #先LDF，后LPF，顺序不能乱
        matricesLDFLPF.append(matrixLDFLPF.tolist()) #tolist: Numpy array to list
        
        matrixRad = []
        with open(os.path.join(directoryPathRad, sFileRad), 'r') as rRad:
            #matrix = [[float(num) for num in line.split()] for line in f]
            matrixRad = np.loadtxt(rRad)
            
            #将Rad进行模糊化
            matrixRad[0,:] *= 2
            matrixRad[-1,:] *= 2
            matrixRad[:,0] *= 2 
            matrixRad[:,-1] *= 2
            matrixRad = gaussian_filter(matrixRad, 3, mode = 'nearest')
            
# =============================================================================
#             #将256*256的矩阵转化为128*128            
#             new_shape = (128, 128)
#             new_matrix = np.zeros(new_shape)
#             for k in range(new_shape[0]):
#                 for j in range(new_shape[1]):
#                     # 计算相邻四个像素的平均值
#                     average = np.mean(matrixRad[k*2:k*2+2, j*2:j*2+2])
#                     # 将平均值存入新矩阵
#                     new_matrix[k, j] = average
#             matrixRad = new_matrix.reshape(1, 128, 128)
# =============================================================================
            matrixRad = matrixRad.reshape(1, matrixSize, matrixSize)
            matricesRad.append(matrixRad.tolist()) #tolist: Numpy array to list

print("Load data successful...")

x = np.stack(matricesLDFLPF, axis=0)  
y = np.stack(matricesRad, axis=0)
del matricesLDFLPF
del matricesRad
#x = pickle.load(open("./data/dataLDF.pkl", "rb"))
#y = pickle.load(open("./data/dataRad.pkl", "rb"))
x = paddle.to_tensor(x, dtype="float32")
y = paddle.to_tensor(y, dtype="float32")

print("Predicting...")
start_time = time.time()
# 测试训练模型

model.eval()
# out = model(x)
with paddle.no_grad(): #不用这个的话，内存会爆炸。但是不用的话效果好像会好一些
    out = model(x)
end_time = time.time()

# 计算代码执行所消耗的时间
elapsed_time = end_time - start_time
ave_time = elapsed_time/selectPredictNum
print("代码执行耗时：{:.5f}秒".format(elapsed_time))
print("平均预测耗时：{:.5f}秒".format(ave_time))
# 计算残差
# error = paddle.abs((out[:, 0, :, :] - y[:, 0, :, :]) / y[:, 0, :, :])
error = (out[:, 0, :, :] - y[:, 0, :, :]) / y[:, 0, :, :]
error = paddle.unsqueeze(error, axis=1) #插入维度，50*192*192 -> 50*1*192*192

# 作出CFD和CNN的计算结果对比图以及对应的残差图(s可修改)


plt.rcParams['font.sans-serif']=['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'custom'  
mpl.rcParams['mathtext.rm'] = 'Times New Roman'

meanError = []
stdError = []
# xianyanbao = 157
# visualize(y.detach().numpy(), out.detach().numpy(), error.detach().numpy(), xianyanbao, sampleNumLDF[random_indices[xianyanbao]])
# plotField(x.detach().numpy(), 'LDF', xianyanbao, sampleNumLDF[random_indices[xianyanbao]])
# plotField(x.detach().numpy(), 'LPF', xianyanbao, sampleNumLDF[random_indices[xianyanbao]])
# plotField(y.detach().numpy(), 'Rad', xianyanbao, sampleNumLDF[random_indices[xianyanbao]])
# plotField(out.detach().numpy(), 'CNN', xianyanbao, sampleNumLDF[random_indices[xianyanbao]])

x = x.detach().numpy()
y = y.detach().numpy()
out = out.detach().numpy()
error = error.detach().numpy()
for s in range (0, selectPredictNum) :#第几号样本
    # visualize(y, out, error, s, sampleNumLDF[random_indices[s]])
    # plotField(x, 'LDF', s, sampleNumLDF[random_indices[s]])
    # plotField(x, 'LPF', s, sampleNumLDF[random_indices[s]])
    # plotField(y, 'Rad', s, sampleNumLDF[random_indices[s]])
    # plotField(out, 'CNN', s, sampleNumLDF[random_indices[s]])
    meanError.append(np.mean(np.abs(error[s,0,:,:])))
    stdError.append(np.std(error[s,0,:,:]))
    
del x
del y
del out
del error
meanError = np.array(meanError)
aveMeanError = np.average(meanError)
stdError = np.array(stdError)
aveStdError = np.average(stdError)
print("meanError = ", aveMeanError) 
print("stdError = ", aveStdError)
print("std of MeanError = ", np.std(meanError))
count_high_error = sum(1 for num in meanError if num > 0.06)
print("meanError > 0.06: num = ", count_high_error, "; percentage = ", count_high_error / selectPredictNum)

count_dict = {}
for s in range (0, selectPredictNum) :
    index = random_indices[s]
    if meanError[s] > 0.06:
        print("index =", s, ",\tsampleNum =", sampleNumLDF[index], ",\tval =", meanError[s])
        first_int = int(sampleNumLDF[index].split('_')[0])
        if first_int in count_dict:
            count_dict[first_int] += 1
        else:
            count_dict[first_int] = 1
count_dict = dict(sorted(count_dict.items()))
for key, value in count_dict.items():
    print(f" {key} ：{value}")

sort_indices = np.argsort(meanError) #按照meanError排序
meanError = meanError[sort_indices]
stdError  = stdError[sort_indices]

from matplotlib import pyplot as plt
x = np.linspace(0, meanError.shape[0], meanError.shape[0])
fig, ax = plt.subplots(figsize=(15, 10))
# plt.title("MODEL #" + str(modelNum), fontsize=32)
plt.title(" FRPM Performance", fontsize=40, pad=15)
# plt.text(-selectPredictNum/20.0, np.max(meanError)*1.02, "#" + str(modelNum) , color='#8c8c8c',fontsize=19)
plt.suptitle(r"$\mathit{t_{\mathregular{ave}}}$" + "= {:.3f}s".format(ave_time), fontsize=28, x=0.82, y=0.92)
# plt.suptitle(r"$\mathit{t_{\mathregular{ave}}}$" + "= 0.095s", fontsize=28, x=0.82, y=0.92)
# plt.plot(x, meanError, label=r"$\mathregular{absError_{ave}}$" + " = {:.2%}".format(aveMeanError), linestyle = '-',linewidth = 1.5)
# plt.plot(x, stdError, label=r"$\mathregular{stdError_{ave}}$" + " = {:.2%}".format(aveStdError), marker ='.',linestyle='None' ,linewidth = 2)
bar = ax.bar(x, meanError, label=r"$\mathregular{\mu(\vert{\mathit{E}}\vert)}$ ,  " + r"$\mathregular{\mu}$"+" = {:.2%}".format(aveMeanError), color='#2e5fa1',width = 1.1)
line, = ax.plot(x, stdError, label=r"$\mathregular{\sigma({\mathit{E}})}$   ,  "+ r"$\mathregular{\mu}$"+ " = {:.2%}".format(aveStdError), marker ='.',linestyle='None',color='#e45007' ,linewidth = 1.2)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.1%}'.format(x)))
plt.xlabel('Samples', fontsize=32)
plt.ylabel('Percentage Error', fontsize=32)
plt.grid(True, which='major', linestyle=':',color='0',axis='y',linewidth=2,alpha=0.3)
# ax.set_ylim(-0.00, np.max(meanError))
ax.set_ylim(-0.00, np.max(meanError))
ax.tick_params(axis='x', direction='in', length=6, width=1.5)
ax.tick_params(axis='y', direction='in', length=6, width=1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.text(-selectPredictNum/20.0, aveMeanError+0.0016, "{:.2%}".format(aveMeanError), color='#2e5394',fontsize=19)
plt.text(-selectPredictNum/20.0, aveStdError-0.0035, "{:.2%}".format(aveStdError), color='#e45007',fontsize=19)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.axhline(y= aveMeanError, color='#2e5394',linestyle='--',linewidth = 2.5)
plt.axhline(y= aveStdError, color='#e45007',linestyle=':',linewidth = 3.5)
handles = [bar, line]
labels = [h.get_label() for h in handles]
legend = ax.legend(handles, labels, fontsize=28,shadow=1)
for text in legend.get_texts():
    text.set_text(text.get_text() + "  ")  # 添加空格以保持对齐
    text.set_ha("right")
plt.show()


