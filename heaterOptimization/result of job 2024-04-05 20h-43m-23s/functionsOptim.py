'''
@author: Hongjie Xing
'''
from matplotlib import pyplot as plt
from matplotlib import rc
import os
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
import numpy as np
import datetime

def generateHeaterInfo(numLamps, sampleNum, receiverYdis, lampInfo):    
    retInfo = f"{numLamps} {sampleNum} {receiverYdis:.4f} "
    for arr in lampInfo:
        formatArr = " ".join([f"{x:.4f}" for x in arr])
        retInfo += (formatArr + " ")
    return retInfo

def plotFieldWithBox(Field, _halfWidth, mode='LPF', info=''):
    plt.rcParams['font.sans-serif']=['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'custom'  
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
        
    plt.figure(figsize=(8,6)) # dpi*figsize=pixel,
    minField = np.min(Field)
    maxField = np.max(Field)
    if mode == 'Rad':
        Field = gaussian_filter(Field, 3, mode = 'nearest')
    plt.imshow(Field, cmap=plt.cm.jet, vmin=minField, vmax=maxField, origin='lower',
           extent=[0, 1000, 0, 1000], interpolation=None, zorder=1)
    # _start = int(ROW/2-_halfWidth)
    # _end = int(ROW/2+_halfWidth)
    _widthz = 500
    _widthx = 500
    _startz = 500 - _widthz//2
    _startx = 500 - _widthx//2
    # center = Field[_start : _end, _start : _end]
    # if mode == 'Rad':
    #     edgecolor = '#ffffff'
    # else:
    #     edgecolor = '#ffffff'
    rect = Rectangle((_startx, _startz),_widthx, _widthz, linewidth=2,linestyle='-', edgecolor='#ffffff', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    
    #plt.clim(0.9,2)
    # hide the axis label
    plt.xticks([0,250,500,750,1000],fontsize=14)
    plt.yticks([0,250,500,750,1000],fontsize=14)
    # Set the x and y axis labels
    plt.xlabel('x Position / mm', fontsize=22)
    plt.ylabel('z Position / mm', fontsize=21)
    
    cb = plt.colorbar(cax=None,ax=None,shrink=1.0, format ="%-5.2f",ticks=np.linspace(Field.min(), Field.max(), 5))
    if mode == 'Rad':
        mode = 'Heat Flux / $\\mathregular{kW\cdot m^{-2}}$'
        ptitle = 'RDM  '
        # plt.suptitle(info, fontsize=16, x=0.2) 
        plt.text(-250, 1050, info, fontsize=16)
    else:
        mode += ' Level'
        ptitle = 'LPF  '
        # plt.suptitle(info, fontsize=16, x=0.05) 
        plt.text(-250, 1050, info, fontsize=16)
    plt.title(ptitle, fontsize=20)
    cb.ax.set_title(mode, fontsize=12, pad=6)
    # cb.ax.set_title('LPF Level ($\\mathregular{mm}$)', fontsize=14, pad=10)
    cb.ax.tick_params(labelsize=12)
    
    
    #plt.subplots_adjust(left=0,right=7,bottom=1,top=5)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.subplots_adjust(left=0.14,right=0.94,bottom=0.1,top=0.9)
    import time
    plt.savefig(f'./saveFig/{current_time}.png', dpi = 200)
    time.sleep(1)
    
    # plt.show()
    plt.close()
    #plt.clf()
    
def calcuSubField(Field, _halfWidth, mode='all'):
    ROW = Field.shape[0]
    COL = Field.shape[1]
    _bottom = int(ROW//2 - _halfWidth)
    _top    = int(ROW//2 + _halfWidth)
    _left   = int(COL//2 - _halfWidth)
    _right  = int(COL//2 + _halfWidth)
    if mode == 'long': # z:375-625(72-120) x:125-875(24-168), 
        sub_field = Field[72 : 120, 24 : 168]
    else:
        sub_field = Field[_bottom : _top, _left : _right] #[,)左包右不包

    ''' Min-Max Scaling before calculate std value '''
    if mode == 'vertical':
        _sub = Field[_bottom : _top, COL//2]
        _min = np.min(_sub)
        _max = np.max(_sub)
        retAve = np.average(_sub)
        retStd = np.std((_sub - _min) / (_max - _min))
    elif mode == 'horizontal':
        _sub = Field[ROW//2, _left : _right]
        _min = np.min(_sub)
        _max = np.max(_sub)
        retAve = np.average(_sub)
        retStd = np.std((_sub - _min) / (_max - _min))
    else:
        _min = np.min(sub_field)
        _max = np.max(sub_field)
        retAve = np.average(sub_field)
        # retStd = np.std((sub_field - _min) / (_max - _min)) #加减某个常数，不影响结果
        retStd = np.std((sub_field - _min) / retAve) #L1-normalization = sub_field / sum(abs(sub_field))
        
    return sub_field, retAve, retStd

''' NSGA-II '''
import configparser
import re
import operator
import ctypes
from RIMNet.UNetEx import *

''' Prepare dataset 1: infos '''
''' numLamps sampleNum receiverYdis (lampL x y z a1 a2)*numLamps '''
# numLamps: 单层石英灯数！
def generateSystemInfo(numLamps, lampL, lampSpace, receiverYdis, layer=1, isStagger=0, layerSpace=0, 
                       dx = 0.0, dy = 0.0, anglez2x=0.0, anglez2y=0.0):
    sampleNum = 0 #no change
    numLamps = np.array(numLamps).astype(int)
    layer = np.array(layer).astype(int)
    retInfo = []
    
    for i, num in enumerate(numLamps):
        lampInfo = []
        xn = changeXpos(numLamps=num, lampSpace=lampSpace[i], dx = dx[i]) #numLamps个
        yn = changeYpos(numLamps=num, dy = dy[i])
        assert(num > 2)
        assert(layer[i] < 3)
        if layer[i] == 2:
            for j in range (0, num):
                yn.append(yn[j]) #第二层
            if isStagger == 0:  # if isStagger[i] == 0:
                for j in range (0, num):
                    xn.append(xn[j])
            else:
                for j in range (0, num-1):
                    xn.append((xn[j]+xn[j+1])*0.5)
                xn.append(2 * xn[num-1] - xn[-1])
        dx_center = (xn[-1] + xn[0]) * 0.5 #用于居中
        for j in range (0, num*layer[i]):
            x = xn[j] - dx_center
            y = yn[j]
            if layer[i] == 1:
                z = 0
            else:
                z = (layerSpace[i] + lampL[i]) * 0.5 * (1 - j // num * 2)
            angle_z2x, angle_z2y = anglez2x, anglez2y
            lampInfo.append( [lampL[i], x, y, z, angle_z2x, angle_z2y] )
        retInfo.append( generateHeaterInfo(num*layer[i], sampleNum, receiverYdis[i], lampInfo) )
        
    return retInfo

def changeXpos(numLamps, lampSpace, dx=0.0): #dx[0,1], numLamps[3,12]
    dx *= lampSpace
    # print(numLamps)
    if numLamps > 7:
        dx *= (numLamps * 0.2)
    _start = -(numLamps * 0.5 - 0.5) * lampSpace
    xn = [_start + i * lampSpace for i in range(int(numLamps))]
    xn[numLamps//2-1]          -= dx * 0.5
    xn[numLamps//2+numLamps%2] += dx * 0.5
    if numLamps > 3:
        newLampSpace = abs(xn[numLamps//2-1] - xn[0]) / (numLamps//2-1)
        for i in range (1, numLamps//2-1):
            xn[i] = xn[0] + i * newLampSpace
    for i in range (0, numLamps//2):
        xn[-1-i] = -xn[i]
    return xn

'''第1种，用于bestShow
def changeYpos(numLamps, dy=0.0): #dy[0,1], numLamps[3,12]，需要确保anglez2y==0
    dy -= 0.3 #[-0.3,0.7]
    dy *= 180.0 #[-54, 126]mm
    yn = [dy] * numLamps
    assert(numLamps > 2)
    ddy = dy / (numLamps//2 - 1 + numLamps%2) #3-1,4-1,5-2,6-2
    for i in range (0, numLamps//2):
        yn[i] -= ddy * i
        yn[-1-i] = yn[i]
    if numLamps%2 == 1:
        yn[numLamps//2] = 0
    # yn -= ((max(yn) - min(yn))/2.0)
    return yn
'''

def changeYpos(numLamps, dy=0.0): #dy[0,1], numLamps[3,12]，需要确保anglez2y==0
    #dy -= 0.3 #[-0.3,0.7]
    dy *= 200.0 #[-54, 126]mm
    yn = [dy] * numLamps
    assert(numLamps > 2)
    ddy = dy / (numLamps//2 - 1 + numLamps%2) #3-1,4-1,5-2,6-2
    for i in range (0, numLamps//2):
        yn[i] -= ddy * i
        yn[-1-i] = yn[i]
    if numLamps%2 == 1:
        yn[numLamps//2] = 0
    yn -= ((max(yn) - min(yn))/2.0)
    return yn.tolist()


''' Prepare dataset 2: calculate LDF&LPF by dll '''
def calcuLDFLPF(dll, LDFptr, LPFptr, heaterInfo, numLamps, ROW=192, COL=192, halfWidth=48):  
    # print(len(heaterInfo)) #50
    matricesLDFLPF = []
    for i, info in enumerate(heaterInfo):
        dll.fieldConstructor()
        dll.createHeater.argtypes = [ctypes.c_char_p, ctypes.c_int]
        dll.createHeater(info.encode(), int(numLamps[i]))
        if not dll.runLDFandLPF(): # run func dll.runLDFandLPF() and check 
            print("LDF & LPF are nullptr.")
        matrixLDF = np.copy(np.ctypeslib.as_array(LDFptr, shape=(ROW, COL))) #不用np.copy将会导致随机。copy后需调用c++中的deleteField函数释放空间
        matrixLPF = np.copy(np.ctypeslib.as_array(LPFptr, shape=(ROW, COL)))
        dll.deleteField()
        matrixLDFLPF = np.stack((matrixLDF[::-1], matrixLPF[::-1]), axis=0) #先LDF，后LPF，顺序不能乱。左右翻转
        matricesLDFLPF.append(matrixLDFLPF.tolist()) #tolist: Numpy array to list
        # plotFieldWithBox(matrixLPF, halfWidth, info = r"d$\mathit{x}$ = "+str(spaces[i]) + ' mm')
    return matricesLDFLPF

def predictRDM(model, matricesLDFLPF): #numpy array
    x = np.stack(matricesLDFLPF, axis=0)  
    del matricesLDFLPF
    x = paddle.to_tensor(x, dtype="float32")
    model.eval()
    with paddle.no_grad(): #不用这个的话，内存会爆炸。但是不用的话效果好像会好一些
        out = model(x)
    out = out.detach().numpy()
    return out

def analyzeRDM(RDM, P):
    efficiency=[]
    uniformity=[]
    ROW = RDM.shape[2]
    COL = RDM.shape[3]
    assert(ROW == 192)
    _halfWidth = ROW//4
    _bottom = int(ROW//2 - _halfWidth)
    _top    = int(ROW//2 + _halfWidth)
    _left   = int(COL//2 - _halfWidth)
    _right  = int(COL//2 + _halfWidth)
    S = (_top - _bottom) * (_right - _left) * (1/ROW) * (1/COL) # m^2
    for i in range(0, RDM.shape[0]):
        sub_field = RDM[i, 0, _bottom : _top, _left : _right] #[,)左包右不包
        mean=np.mean(sub_field)
        efficiency.append(mean * S / P[i])
        uniformity.append(np.std(sub_field/mean))
    return efficiency, uniformity


import geatpy as ea
class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, dll, model):
        name = 'bestHeater' # 初始化name（函数名称，可以随意设置）
        M = 2 # 初始化M（目标维数）
        Dim = 8 # 初始化Dim（决策变量维数）
        maxormins = [-1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        #         f(n,  L,    s1,  s2,   m, s3,  dx, dy)
        varTypes = [1,  1,    0,   0,    1, 0,   0,  0] # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb =       [3,  150,  20,  150,  1, 0,   0,  0] # 决策变量下界
        ub =       [12, 1000, 200, 1000, 2, 100, 1,  1] # 决策变量上界ub_s3=200, m=1时ub_s3=0
        lbin =     [1,  1,    1,   1,    1, 1,   1,  1] # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin =     [1,  1,    1,   0,    1, 1,   1,  1] # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin=lbin, ubin=ubin)
        self.dll = dll
        self.model = model

    def evalVars(self, Vars): # 目标函数
        n, L, s1, s2, m, s3, dx, dy = Vars[:,0], Vars[:,1], Vars[:,2], Vars[:,3], Vars[:,4], Vars[:,5], Vars[:,6], Vars[:,7]
        
        # ROW = ctypes.c_int.in_dll(dll, 'row').value #取值，dll.row是地址;ctypes.c_int.in_dll(dll, 'row')表示int，长度为4
        # COL = ctypes.c_int.in_dll(dll, 'col').value 
        LDFptr = ctypes.POINTER(ctypes.c_double).in_dll(self.dll, 'LDF')#print(ctypes.sizeof(LDFField_ptr))：8，double*
        LPFptr = ctypes.POINTER(ctypes.c_double).in_dll(self.dll, 'LPF')
        # halfWidth = ROW//4
        heaterInfo = generateSystemInfo(n, L, s1, s2, layer=m, isStagger=0, layerSpace=s3, 
                               dx = dx, dy = dy, anglez2x=0.0, anglez2y=0.0)
        # print(heaterInfo, n*m)
        matrixLDFLPF = calcuLDFLPF(self.dll, LDFptr, LPFptr, heaterInfo, int(n*m), ROW=192, COL=192, halfWidth=48)
        RDM = predictRDM(self.model, matrixLDFLPF)
        P = n * L * m * 3.75 * 0.001 #P [kW]
        efficiency, uniformity = analyzeRDM(RDM = RDM, P = P)
        # print(max(efficiency),', ',min(uniformity))
        efficiency = np.array(efficiency).reshape(-1,1)
        uniformity = np.array(uniformity).reshape(-1,1)
        CV =  np.hstack([efficiency-0.8, uniformity-0.5]) #efficiency-1<=0, -uniformity<=0 两个目标 两列
        ObjV = np.hstack([efficiency, uniformity]) # 计算目标函数值，赋值给pop种群对象的ObjV属性
        return ObjV, CV

