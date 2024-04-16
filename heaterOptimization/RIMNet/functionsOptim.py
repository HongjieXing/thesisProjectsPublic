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

def plotFieldWithBox(Field, _halfWidth, mode='LPF', info='', zcenter = 0):
    plt.rcParams['font.sans-serif']=['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'custom'  
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
        
    plt.figure(figsize=(8,6)) # dpi*figsize=pixel,
    minField = np.min(Field)
    maxField = np.max(Field)
    plt.imshow(Field, cmap=plt.cm.jet, vmin=minField, vmax=maxField, origin='lower',
           extent=[0, 1000, 0, 1000], interpolation=None, zorder=1)
    # _start = int(ROW/2-_halfWidth)
    # _end = int(ROW/2+_halfWidth)
    # [12 : 180, 72 : 120]
    _widthz = 875
    _widthx = 250
    _startz = 500 - _widthz//2
    _startx = 500 - _widthx//2
    if (zcenter > 0) and (zcenter < 490):
        _startz = 500 - zcenter
        _widthz = 2 * zcenter
    elif zcenter >= 490:
        zcenter = 490
        _startz = 500 - zcenter
        _widthz = 2 * zcenter
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
    else:
        mode += ' Level'
        ptitle = 'LPF  '
    plt.title(ptitle + info, fontsize=20)
    cb.ax.set_title(mode, fontsize=12, pad=6)
    # cb.ax.set_title('LPF Level ($\\mathregular{mm}$)', fontsize=14, pad=10)
    cb.ax.tick_params(labelsize=12)
    
    
    #plt.subplots_adjust(left=0,right=7,bottom=1,top=5)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.subplots_adjust(left=0.14,right=0.94,bottom=0.1,top=0.9)
    import time
    # plt.savefig(f'./saveFig/{current_time}.png', dpi = 200)
    # time.sleep(1)
    
    plt.show()
    # plt.close()
    #plt.clf()
    
def calcuSubField(Field, _halfWidth, mode='all',zcenter=0):
    ROW = Field.shape[0]
    COL = Field.shape[1]
    _bottom = int(ROW//2 - _halfWidth)
    _top    = int(ROW//2 + _halfWidth)
    _left   = int(COL//2 - _halfWidth)
    _right  = int(COL//2 + _halfWidth)
    if mode == 'long': # z:375-625(72-120) x:125-875(24-168), 
        sub_field = Field[72 : 120, 24 : 168]
    elif mode =='vertical':
        sub_field = Field[12 : 180, 72 : 120]
    else:
        sub_field = Field[_bottom : _top, _left : _right] #[,)左包右不包

    ''' Min-Max Scaling before calculate std value '''
    if mode == 'vertical':
        _sub = sub_field
        _min = np.min(_sub)
        _max = np.max(_sub)
        retAve = np.average(_sub)
        if zcenter < 500.0:
            dz = int(zcenter / 1000.0 * 192)
        else:
            dz = 95
        _ave = np.average(Field[95-dz:95+dz, 72:120])
        # retStd = _ave / Field[95, 95] #1
        retStd = _max / Field[95, 95] #2 比较好，表征最大值和中心值的比值
        # retStd = abs(retStd-1) #是否更接近于1，作图效果不大
        # print(retStd)
        # retStd = retAve / (Field[ROW//2, COL//2])
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




