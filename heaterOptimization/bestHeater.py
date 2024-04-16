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
    # ROW = ctypes.c_int.in_dll(dll, 'row').value #取值，dll.row是地址;ctypes.c_int.in_dll(dll, 'row')表示int，长度为4
    # COL = ctypes.c_int.in_dll(dll, 'col').value 
    # LDFField_ptr = ctypes.POINTER(ctypes.c_double).in_dll(dll, 'LDF')#print(ctypes.sizeof(LDFField_ptr))：8，double*
    # LPFField_ptr = ctypes.POINTER(ctypes.c_double).in_dll(dll, 'LPF')
    # halfWidth = ROW//4
    
    ''' run '''
    problem = MyProblem(dll, model)
    myAlgorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='RI', NIND=50),
        MAXGEN=1000,  
        logTras=10)   
    myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    # 求解
    res = ea.optimize(myAlgorithm,
                      seed = 42, 
                      verbose=True,
                      drawing=1,# 0表示不绘图; 1表示绘制最终结果图; 2表示实时绘制目标空间动态图; 3表示实时绘制决策空间动态图。
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True)
    print(res)
    
    

    '''    卸载DLL    '''
    free_library = ctypes.WinDLL('kernel32', use_last_error=True).FreeLibrary
    free_library.argtypes = [ctypes.c_void_p]
    if not free_library(dll._handle):
        print("DLL卸载失败")

# ''' Configure FRPM parameters '''
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# config = configparser.ConfigParser()
# config.read("./RIMNet/config.ini")
# kernel_size = int(config["net_parameter"]["kernel_size"])
# filters = [int(i) for i in config["net_parameter"]["filters"].split(",")]
# bn = bool(int(config["net_parameter"]["batch_norm"]))
# wn = bool(int(config["net_parameter"]["weight_norm"]))
# model = UNetEx(2, 1, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
# model.set_state_dict( paddle.load('./RIMNet/FRPM_9861.pdparams') )


# ''' Load DLL, get const values and set '''
# dll = ctypes.PyDLL("./RadFieldDLL.dll")
# ROW = ctypes.c_int.in_dll(dll, 'row').value #取值，dll.row是地址;ctypes.c_int.in_dll(dll, 'row')表示int，长度为4
# COL = ctypes.c_int.in_dll(dll, 'col').value 
# LDFField_ptr = ctypes.POINTER(ctypes.c_double).in_dll(dll, 'LDF')#print(ctypes.sizeof(LDFField_ptr))：8，double*
# LPFField_ptr = ctypes.POINTER(ctypes.c_double).in_dll(dll, 'LPF')
# halfWidth = ROW//4


# ''' Prepare dataset 1: infos '''
# ''' numLamps sampleNum receiverYdis (lampL x y z a1 a2)*numLamps '''
# # info = "3 674 330.0000 150.0000 -289.0000 0.0000 185.0000 0.2618 0.0000 150.0000 0.0000 0.0000 -185.0000 0.2618 0.0000 150.0000 289.0000 0.0000 185.0000 0.2618 0.0000"
# numLamps = 9
# lampSpace = 250
# sampleNum = 0 #no change
# receiverYdis = 100
# lampL = 150
# infos = []

# stepVal = 10.0
# stepNum = 60
# spaces = np.arange(10.0, 10.0 + stepVal * stepNum, stepVal) #(start, start + step * num_values, step)
# # spaces = [75, 225]
# for space in spaces:
#     lampInfo = []
#     for n in range (0, numLamps):
#         x = -numLamps//2 * lampSpace + n * lampSpace
#         y, z = 0.0, 0.0
#         angle_z2x, angle_z2y = 0.0, 0.0
#         linfo = [lampL, x, y, z, angle_z2x, angle_z2y]
#         lampInfo.append(linfo)
#     receiverYdis = space
#     infos.append( generateHeaterInfo(numLamps, sampleNum, receiverYdis, lampInfo) )


# ''' Prepare dataset 2: calculate LDF&LPF by dll '''
# matricesLDFLPF = []
# for i, info in enumerate(infos):
#     # dll.printField(info)
#     dll.fieldConstructor()
#     dll.createHeater.argtypes = [ctypes.c_char_p, ctypes.c_int]
#     dll.createHeater(info.encode(), numLamps)
#     # dll.runLDFandLPF.restype = ctypes.c_bool #bool值可以视为int
#     # resLDFLPF = dll.runLDFandLPF()
#     if not dll.runLDFandLPF(): # run func dll.runLDFandLPF() and check 
#         print("LDF & LPF are nullptr.")
#     # print( "LDF & LPF are calculated successfully." if resLDFLPF else "LDF & LPF are nullptr.")
#     matrixLDF = np.copy(np.ctypeslib.as_array(LDFField_ptr, shape=(ROW, COL))) #不用np.copy将会导致随机。copy后需调用c++中的deleteField函数释放空间
#     matrixLPF = np.copy(np.ctypeslib.as_array(LPFField_ptr, shape=(ROW, COL)))
#     dll.deleteField()
    
#     matrixLDFLPF = np.stack((matrixLDF[::-1], matrixLPF[::-1]), axis=0) #先LDF，后LPF，顺序不能乱
#     matricesLDFLPF.append(matrixLDFLPF.tolist()) #tolist: Numpy array to list
#     # plot LPF
#     # plotFieldWithBox(matrixLPF, halfWidth, info = r"d$\mathit{x}$ = "+str(spaces[i]) + ' mm')


# ''' Predict '''
# x = np.stack(matricesLDFLPF, axis=0)  
# del matricesLDFLPF
# x = paddle.to_tensor(x, dtype="float32")
# model.eval()
# with paddle.no_grad(): #不用这个的话，内存会爆炸。但是不用的话效果好像会好一些
#     out = model(x)
# out = out.detach().numpy()


# ''' Analyze and Plot '''
# stdValues = []
# aveValues = []
# for i in range(0, out.shape[0]):
#     subField, aveField, stdField = calcuSubField(out[i, 0, :, :], halfWidth, mode='all')
#     stdValues.append(stdField)
#     aveValues.append(aveField)
#     # print(f"{spaces[i]:.2f}: mean = {aveField:.4f}, std={stdField:.4f}")
#     # plot Rad:
#     # plotFieldWithBox(out[i, 0, :, :], halfWidth, mode='Rad', info = r"d$\mathit{x}$ = "+str(spaces[i]) + ' mm')

# # stdValues = [1/x for x in stdValues]

# stdValues = (stdValues - min(stdValues)) / (max(stdValues) - min(stdValues))
# stdValues = -(stdValues - 1.0)
# aveValues = (aveValues - min(aveValues)) / (max(aveValues) - min(aveValues))

# '''    卸载DLL    '''
# free_library = ctypes.WinDLL('kernel32', use_last_error=True).FreeLibrary
# free_library.argtypes = [ctypes.c_void_p]
# if not free_library(dll._handle):
#     print("DLL卸载失败")
    


# ''' Plot result '''
# import datetime
# from matplotlib import pyplot as plt
# import matplotlib as mpl
# from scipy.interpolate import interp1d
# plt.rcParams['font.sans-serif']=['Times New Roman']
# mpl.rcParams['mathtext.fontset'] = 'custom'  
# mpl.rcParams['mathtext.rm'] = 'Times New Roman'

# fig, ax = plt.subplots(figsize=(15, 10))
# ax2 = ax.twinx()
# # plt.title("MODEL #" + str(modelNum), fontsize=32)
# # plt.title(f'Heater-Receiver Distance = {receiverYdis} mm', fontsize=35, pad=15)
# ax.scatter(spaces, stdValues, c='#2e5fa1', marker='o', linewidths=2.0, edgecolors='#343a40', s=150, alpha=0.8, label='Uniformity')
# ax.scatter(spaces, aveValues, c='#e45007', marker='v', linewidths=2.0, edgecolors='#343a40', s=180, alpha=0.8, label='Efficiency')

# # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.1%}'.format(x)))  #设置y轴坐标为{:.1%}

# ax.set_xlabel('Heater-Receiver Distance' + ' / mm', fontsize=34)
# ax.set_ylabel('Uniformity', color='#2e5fa1', fontsize=34)
# ax2.set_ylabel('Efficiency', color='#e45007', fontsize=34)
# ax.tick_params(axis='x', labelsize=28)
# ax.tick_params(axis='y', labelcolor='#2e5fa1', labelsize=28)
# ax2.tick_params(axis='y', labelcolor='#e45007', labelsize=28)
# # ax.set_ylim(min(stdValues)*0.9, max(stdValues)*1.03)
# ax.set_ylim(-0.05, 1.05)
# ax2.set_ylim(-0.05, 1.05)
# ax.tick_params(axis='x', direction='in', length=8, width=2)
# ax.tick_params(axis='y', direction='in', length=8, width=2)
# ax2.tick_params(axis='y', direction='in', length=8, width=2)
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
# plt.xticks(fontsize=28)
# plt.yticks(fontsize=28)
# # plt.grid(True, which='major', linestyle=':',color='0',axis='y',linewidth=2,alpha=0.3)
# legend = ax.legend(fontsize=28, markerscale=1.5,labelcolor='black',shadow=1)

# # legend = ax2.legend(fontsize=28, markerscale=1.5,labelcolor='black',shadow=1)
# # plt.axhline(y = 0.086, color='#e45007',linestyle='--',linewidth = 1.5)
# # plt.axvline(x = 190, linestyle='--', color='#008d00', linewidth = 1)
# # plt.axvline(x = 260, linestyle='--', color='#008d00', linewidth = 1)
# # plt.axvline(x = 380, linestyle='--', color='#008d00', linewidth = 1)
# # plt.text(195, max(stdValues)*0.95, r"$\mathit{l}$ = 190", color='#008d00',fontsize=20)
# # plt.text(265, max(stdValues)*0.95, r"$\mathit{l}$ = 260", color='#008d00',fontsize=20)
# # plt.text(385, max(stdValues)*0.95, r"$\mathit{l}$ = 380", color='#008d00',fontsize=20)
# # 样条插值
# # f = interp1d(lampLs, stdValues, kind='cubic')  # 选择插值方法为三次样条插值
# # x_smooth = np.linspace(lampLs.min(), lampLs.max(), 400)  # 创建更密集的x值
# # y_smooth = f(x_smooth)  # 计算插值得到的光滑曲线的y值
# # plt.plot(x_smooth, y_smooth, label='Spline Interpolation',color='#2b579a',linewidth = 2)
# current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# plt.savefig(os.path.join("./saveFig/" + f'{current_time}.png'), dpi = 200)
# plt.show()



