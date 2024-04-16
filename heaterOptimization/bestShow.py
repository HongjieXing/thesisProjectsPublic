'''
cout的内容都在最后打印
dll.xxx 打印的是xxx的地址
my_func3.restype = ctypes.POINTER(ctypes.c_int)返回值为int*，取内容用:my_func3(...).contents.value
ctypes.byref()取引用
@author: Hongjie Xing
'''
import configparser
import os
import re
import operator
from RIMNet.UNetEx import *

from functionsOptim import *
import ctypes
import numpy as np
import geatpy as ea


if __name__ == '__main__':
    
    ''' Configure FRPM parameters '''
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    config = configparser.ConfigParser()
    config.read("./RIMNet/config.ini")
    kernel_size = int(config["net_parameter"]["kernel_size"])
    filters = [int(i) for i in config["net_parameter"]["filters"].split(",")]
    bn = bool(int(config["net_parameter"]["batch_norm"]))
    wn = bool(int(config["net_parameter"]["weight_norm"]))
    model = UNetEx(2, 1, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    model.set_state_dict( paddle.load('./RIMNet/FRPM_9861.pdparams') )
    
    
    ''' Load DLL, get const values and set '''
    dll = ctypes.PyDLL("./RadFieldDLL.dll")
    ROW = ctypes.c_int.in_dll(dll, 'row').value #取值，dll.row是地址;ctypes.c_int.in_dll(dll, 'row')表示int，长度为4
    COL = ctypes.c_int.in_dll(dll, 'col').value 
    LDFField_ptr = ctypes.POINTER(ctypes.c_double).in_dll(dll, 'LDF')#print(ctypes.sizeof(LDFField_ptr))：8，double*
    LPFField_ptr = ctypes.POINTER(ctypes.c_double).in_dll(dll, 'LPF')
    halfWidth = ROW//4
    
    
    ''' Prepare dataset 1: infos '''
    ''' numLamps sampleNum receiverYdis (lampL x y z a1 a2)*numLamps '''
    # info = "3 674 330.0000 150.0000 -289.0000 0.0000 185.0000 0.2618 0.0000 150.0000 0.0000 0.0000 -185.0000 0.2618 0.0000 150.0000 289.0000 0.0000 185.0000 0.2618 0.0000"
    
    import csv
    paras = []
    resultPath = './result of job 2024-04-01 16h-54m-57s/optPop/' 
    with open(resultPath+'Phen.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            #            f(n,  L,    s1,  s2,   m, s3,  dx, dy)
            paras.append([float(row[0]), float(row[1]), float(row[2]),float(row[3]), float(row[4]),float(row[5]),float(row[6]),float(row[7])])
    paras = np.array(paras)
    infos = generateSystemInfo(numLamps = paras[:,0], lampL = paras[:,1], lampSpace = paras[:,2], receiverYdis = paras[:,3],
                               layer=paras[:,4], isStagger=0, layerSpace=paras[:,5], 
                               dx = paras[:,6], dy = paras[:,7], anglez2x=0.0, anglez2y=0.0) 
    objvs = []
    with open(resultPath+'ObjV.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            objvs.append([float(row[0]), float(row[1])]) # eff, uni
    objvs = np.array(objvs)
    
    ''' Prepare dataset 2: calculate LDF&LPF by dll '''
    matricesLDFLPF = []
    numLamps = paras[:,0].astype(int)
    layer = paras[:,4].astype(int)
    for i, info in enumerate(infos):
        # dll.printField(info)
        dll.fieldConstructor()
        dll.createHeater.argtypes = [ctypes.c_char_p, ctypes.c_int]
        dll.createHeater(info.encode(), numLamps[i]*layer[i])
        # dll.runLDFandLPF.restype = ctypes.c_bool #bool值可以视为int
        # resLDFLPF = dll.runLDFandLPF()
        if not dll.runLDFandLPF(): # run func dll.runLDFandLPF() and check 
            print("LDF & LPF are nullptr.")
        # print( "LDF & LPF are calculated successfully." if resLDFLPF else "LDF & LPF are nullptr.")
        matrixLDF = np.copy(np.ctypeslib.as_array(LDFField_ptr, shape=(ROW, COL))) #不用np.copy将会导致随机。copy后需调用c++中的deleteField函数释放空间
        matrixLPF = np.copy(np.ctypeslib.as_array(LPFField_ptr, shape=(ROW, COL)))
        dll.deleteField()
        
        matrixLDFLPF = np.stack((matrixLDF[::-1], matrixLPF[::-1]), axis=0) #先LDF，后LPF，顺序不能乱
        matricesLDFLPF.append(matrixLDFLPF.tolist()) #tolist: Numpy array to list
        ''' plot LPF ''' 
        # if numLamps[i] > 2:
        #     # itext = '#'+str(i) + r" $\mathregular{\mu}$=" + str('{:.2%}'.format(meanError))
        #     plotFieldWithBox(matrixLPF, halfWidth, info = '#'+str(i))
    
    ''' Predict '''
    x = np.stack(matricesLDFLPF, axis=0)  
    del matricesLDFLPF
    x = paddle.to_tensor(x, dtype="float32")
    model.eval()
    with paddle.no_grad(): #不用这个的话，内存会爆炸。但是不用的话效果好像会好一些
        out = model(x)
    out = out.detach().numpy()
    
    
    ''' calculate HF '''
    pp0 = 0.005 #kW/mm
    S = 0.25 #m^2
    HF = objvs[:,0] * paras[:,0] * paras[:,1] * paras[:,4] * pp0 / S #HF = eta*n*m*L/S*p
    
    ''' Plot RDM '''
    # for i in range(0, out.shape[0]):#out.shape[0]
    #     if HF[i] > 20:
    #         itext = '#'+str(i) + r", $\eta$=" + str('{:.2%}'.format(objvs[i,0]))+ r", $\mathregular{\sigma}$=" + str('{:.2%}'.format((objvs[i,1])))
    #         plotFieldWithBox(out[i, 0, :, :], halfWidth, mode='Rad', info = itext) 
    
    bestIndex = []
    for i in range(0, len(HF)):
        if HF[i] > 25 and objvs[i,1]<0.06:
            bestIndex.append(i)
            print (i, objvs[i,0], objvs[i,1], HF[i], pp0* paras[i,0] * paras[i,1])
    
    '''    卸载DLL    '''
    free_library = ctypes.WinDLL('kernel32', use_last_error=True).FreeLibrary
    free_library.argtypes = [ctypes.c_void_p]
    if not free_library(dll._handle):
        print("DLL卸载失败")
    
    ''' Plot Pareto Front Plot '''
    import datetime
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from scipy.interpolate import interp1d
    plt.rcParams['font.sans-serif']=['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'custom'  
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    #ax.scatter(spaces, stdValues, c='#2e5fa1', marker='o', linewidths=2.0, edgecolors='#343a40', s=150, alpha=0.8, label='Uniformity')
    ax.scatter(objvs[:,0], objvs[:,1], c='#1c8041', marker='o', s=150, alpha=0.8, label='PF ' + r"$\mathregular{\sigma}-{\eta}$")
    
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.1%}'.format(x)))  #设置y轴坐标为{:.1%}
    ax.set_xlabel(r"${\eta}$", fontsize=40, color='#2e5fa1')
    ax.set_ylabel(r"$\mathregular{\sigma}$", fontsize=40,color='#c12c1f')
    ax.tick_params(axis='x', labelcolor='#2e5fa1', labelsize=34)
    ax.tick_params(axis='y', labelcolor='#c12c1f', labelsize=34)
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(-0.02, 0.9)
    ax.tick_params(axis='x', direction='in', length=8, width=2)
    ax.tick_params(axis='y', direction='in', length=8, width=2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    plt.grid(True, which='major', linestyle=':',color='0',axis='y',linewidth=2,alpha=0.3)
    plt.grid(True, which='major', linestyle=':',color='0',axis='x',linewidth=2,alpha=0.3)
    plt.axhline(y = 0.06, color='#ae3ec9',linestyle='--',linewidth = 2)
    plt.axvline(x = 0.207, color='#ae3ec9',linestyle='--',linewidth = 2)
    plt.text(-0.04, 0.045, '0.06', fontsize=28,color="#ae3ec9")
    plt.text(0.21, 0.15, '0.207', fontsize=28,color="#ae3ec9")
    ax.scatter(objvs[12,0], objvs[12,1], c='#c00000', marker='o', s=150, alpha=1, label='Selected')
    legend = ax.legend(fontsize=34, markerscale=1.8,labelcolor='black',shadow=1, loc = 'upper left')
    plt.subplots_adjust(left=0.08,right=0.94,bottom=0.12,top=0.96)
    
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(os.path.join("./saveFig/" + f'{current_time}.png'), dpi = 200)
    plt.show()
    import time
    time.sleep(1)
    
    ''' Plot Pareto Front with HF and sigma '''
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(HF, objvs[:,1], c='#1c8041', marker='v', s=150, alpha=0.8, label='PF '+r"$\mathregular{\sigma}-HF$")
    
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.1%}'.format(x)))  #设置y轴坐标为{:.1%}
    ax.set_xlabel('Peak Mean HF / $\\mathregular{kW\cdot m^{-2}}$', fontsize=40, color='#2e5fa1')
    ax.set_ylabel(r"$\mathregular{\sigma}$", fontsize=40,color='#c12c1f')
    ax.tick_params(axis='x', labelcolor='#2e5fa1', labelsize=34)
    ax.tick_params(axis='y', labelcolor='#c12c1f', labelsize=34)
    # ax.set_xlim(600, 9000)
    ax.set_ylim(-0.02, 0.9)
    ax.tick_params(axis='x', direction='in', length=8, width=2)
    ax.tick_params(axis='y', direction='in', length=8, width=2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    plt.grid(True, which='major', linestyle=':',color='0',axis='y',linewidth=2,alpha=0.3)
    plt.grid(True, which='major', linestyle=':',color='0',axis='x',linewidth=2,alpha=0.3)
    plt.axhline(y = 0.06, color='#ae3ec9',linestyle='--',linewidth = 2)
    plt.axvline(x = 28.5, color='#ae3ec9',linestyle='--',linewidth = 2)
    plt.text(-0.2, 0.045, '0.06', fontsize=28,color="#ae3ec9")
    plt.text(27.5, -0.06, '28.5', fontsize=28,color="#ae3ec9")
    ax.scatter(HF[12], objvs[12,1], c='#c00000', marker='v', s=150, alpha=0.8, label='Selected')
    legend = ax.legend(fontsize=34, markerscale=1.8,labelcolor='black',shadow=1, loc = 'upper right')    
    plt.subplots_adjust(left=0.08,right=0.94,bottom=0.12,top=0.96)
    
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(os.path.join("./saveFig/" + f'{current_time}.png'), dpi = 200)
    plt.show()

    


