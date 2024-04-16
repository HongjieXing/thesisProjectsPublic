# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:20:40 2023

@author: Hongjie Xing

"""

import os
import pickle
import numpy as np
#from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
#import matplotlib as mpl

#selectMode = 'LDFLPF'
selectMode = 'LDFLPF'
matSize = 192
# Create an empty list to store the matrices
matrices = []
directoryPath = "F:/PredictResults/data/trainData/"
directoryPath2 = directoryPath
if   (selectMode == 'Rad') :
    directoryPath += "result_RadField"
elif (selectMode == 'LDF') :
    directoryPath += "result_LDFField"
elif (selectMode == 'LPF') :
    directoryPath += "result_LPFField"
elif (selectMode == 'LDFLPF') :
    directoryPath  += "result_LDFField"
directoryPath2 += "result_LPFField"


i=0
if (selectMode == 'LDFLPF'):
    for LDFfile, LPFfile in zip(os.listdir(directoryPath), os.listdir(directoryPath2)) :
        assert ( LDFfile[3:-1] == LPFfile[3:-1] )
        if LDFfile.endswith('.txt') :
            LDFmatrix = np.loadtxt(os.path.join(directoryPath, LDFfile))
            # LDFmatrix = np.fliplr(LDFmatrix)
        if LPFfile.endswith('.txt') :
            LPFmatrix = np.loadtxt(os.path.join(directoryPath2, LPFfile))
            # LPFmatrix = gaussian_filter(LPFmatrix, sigma=3, mode = 'nearest')
            # LPFmatrix = np.where(LPFmatrix != 0, 150 / LPFmatrix, 0) #将非零数值求倒数，并乘150无量纲化
            # LPFmatrix = np.fliplr(LPFmatrix)
        result_matrix = np.stack((LDFmatrix[::-1], LPFmatrix[::-1]), axis=0) #[::-1]表示最后一个元素开始往前取。上下翻转矩阵需要保证矩阵是二维的[[],[],[]]
        del(LDFmatrix)
        del(LPFmatrix)
        matrices.append(result_matrix.tolist()) #tolist: Numpy array to list
        del result_matrix
        i+=1
        
else : #Rad
    # Loop through each txt file and append the matrix to the list
    for file_name, LPFfile in zip(os.listdir(directoryPath), os.listdir(directoryPath2)) :
        assert ( file_name[3:-1] == LPFfile[3:-1] )
        if file_name.endswith('.txt') :
            matrix = np.loadtxt(os.path.join(directoryPath, file_name))
            if (selectMode == 'LDF'): #需要将LDF进行左右对称操作
                matrix = np.fliplr(matrix)
            if (selectMode == 'LPF'): #需要将LDF进行左右对称操作
                matrix = np.fliplr(matrix)
            if (selectMode == 'Rad'):
                matrix[0,:] *= 2
                matrix[-1,:] *= 2
                matrix[:,0] *= 2 
                matrix[:,-1] *= 2
                # matrix = gaussian_filter(matrix, sigma=3, mode = 'nearest')
            result_matrix = matrix.reshape(1, matSize, matSize)
            matrices.append(result_matrix.tolist()) #tolist: Numpy array to list
            i+=1
            
result = np.stack(matrices, axis=0)
del matrices
print("Size of a tensor = ", result.shape)
# Save the list of matrices as a .pkl file

saveFile = "F:/PredictResults/data/"
if (selectMode == 'Rad') :
    saveFile += "dataRad.pkl"
elif (selectMode == 'LDF') :
    saveFile += "dataLDF.pkl"
elif (selectMode == 'LPF') :
    saveFile += "dataLPF.pkl"
else:
    saveFile += "dataLDFLPF.pkl"
    
with open(saveFile, 'wb') as f:
    pickle.dump(result, f)
    
print("Total number of tensors = ", i)

#x = pickle.load(open("F:/Desktop/ML+CFD/projects/DeepCFD_with_PaddlePaddle/data/LDFandRadData/dataLDF.pkl", "rb"))
#print(x.shape)

