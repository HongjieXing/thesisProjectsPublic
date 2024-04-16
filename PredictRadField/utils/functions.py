import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import os
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter
from scipy.ndimage import gaussian_filter

# 分离张量，用于分离训练集和测试集
def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

# 显示并对比CFD和CNN结果
def visualize(sample_y, out_y, error, s, sampleNumber):    
    minRad1 = np.min(sample_y[s, 0, :, :])
    maxRad1 = np.max(sample_y[s, 0, :, :])
    minRad2 = np.min(out_y[s, 0, :, :])
    maxRad2 = np.max(out_y[s, 0, :, :])
    # minRad = min(minRad1, minRad2)
    # maxRad = max(maxRad1, maxRad2)
    minRad = minRad1
    maxRad = maxRad1
    
    plt.rcParams['font.sans-serif']=['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'custom'  
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    
    plt.figure(figsize=(30, 13))
    plt.suptitle('#' + str(sampleNumber), fontsize=36, x=0.07)  
    #fig = plt.gcf()
    #fig.set_size_inches(24, 16)
    plt.subplot(1, 3, 1)
    plt.title('MCRM', fontsize=48, pad=30)
    plt.imshow(sample_y[s, 0, :, :], cmap=plt.cm.jet, vmin=minRad, vmax=maxRad, origin='lower',
               extent=[0, 1000, 0, 1000], interpolation=None)
    cb = plt.colorbar(orientation='horizontal', format ="%-5.2f",ticks=np.linspace(minRad, maxRad, 5))
    cb.ax.set_title('Heat Flux / $\\mathregular{kW\cdot m^{-2}}$', fontsize=28, pad=15, fontname='Times New Roman')
    cb.ax.tick_params(labelsize=30)
    plt.ylabel('z Position / mm', fontsize=40)
    plt.xlabel('x Position / mm', fontsize=38)
    plt.xticks([0,250,500,750,1000],fontsize=24)
    plt.yticks([0,250,500,750,1000],fontsize=24)
    
    
    plt.subplot(1, 3, 2)
    plt.title('FRPM', fontsize=48, pad=30)
    plt.imshow(out_y[s, 0, :, :], cmap=plt.cm.jet, vmin=minRad, vmax=maxRad, origin='lower',
               extent=[0, 1000, 0, 1000], interpolation='spline16')
    cb = plt.colorbar(orientation='horizontal', format ="%-5.2f",ticks=np.linspace(minRad, maxRad, 5))
    cb.ax.set_title('Heat Flux / $\\mathregular{kW\cdot m^{-2}}$', fontsize=28, pad=15, fontname='Times New Roman')
    cb.ax.tick_params(labelsize=30)
    plt.xlabel('x Position / mm', fontsize=38)
    plt.xticks([0,250,500,750,1000],fontsize=24)
    plt.yticks([0,250,500,750,1000],fontsize=24)
    
    
    stdError  = np.std(error[s,0,:,:])
    error = np.abs(error)
    meanError = np.mean(error[s,0,:,:])
    minERad = np.min(error[s, 0, :, :])
    maxERad = np.max(error[s, 0, :, :])
    
    plt.subplot(1, 3, 3)
    plt.title('Difference in %', fontsize=48, pad=30)
    if (maxERad > 0.5):
        maxERad = 0.5    
    plt.imshow(error[s, 0, :, :], cmap=plt.cm.jet, vmin=0, vmax = maxERad, origin='lower',
               extent=[0, 1000, 0, 1000], interpolation=None)
    
    errorTitle = r"$\mathregular{\mu_{abs}}$ = " + str('{:.2%}'.format(meanError)) + ', ' + r"$\mathregular{\sigma}$ = " + str('{:.2%}'.format(stdError))
    cb = plt.colorbar(orientation='horizontal', format ="%-5.2f",ticks=np.linspace(0, maxERad, 5))
    cb.formatter = StrMethodFormatter("{x:.2%}")  
    cb.update_ticks()
    cb.ax.set_title(errorTitle, fontsize=28, pad=15, fontname='Times New Roman',fontweight='normal', fontstyle='normal')
    cb.ax.tick_params(labelsize=30)
    plt.xlabel('x Position / mm', fontsize=38)
    plt.xticks([0,250,500,750,1000],fontsize=24)
    plt.yticks([0,250,500,750,1000],fontsize=24)
    
    #plt.savefig(os.path.join(os.path.dirname(os.getcwd()) , "PREDICT" , str(s) + ".png"), dpi = 150)
    
    plt.tight_layout()
    #plt.gcf().tight_layout(h_pad=0.9, w_pad=0.2)
    plt.subplots_adjust(left=0.07, bottom=-0.05, right=0.95, top=0.95)
    
    meanError = int(meanError*10000)
    if meanError < 400:
        plt.savefig(os.path.join("./resultFig/" + str(meanError) + '_' + str(sampleNumber) + ".png"), dpi = 100)
    #plt.show()
    plt.close()
    del sample_y
    del out_y
    del error
    
#----------------------------------------#
def plotField(Field, plotMode, s, sampleNumber):
    if plotMode == "LDF":
        savePath = "./resultFig/LDF/"
        ptitle = "LDF"
    elif plotMode == "LPF":
        savePath = "./resultFig/LPF/"
        ptitle = "LPF"
    elif plotMode == "Rad":
        savePath = "./resultFig/Rad/"
        ptitle = "RDM"
    else:
        savePath = "./resultFig/CNN/"
        ptitle = "FRPM"
        
    plt.rc('font',family='Times New Roman') 
        
    if plotMode == "LPF":
        matrix = Field[s, 1, :, :]
        # matrix = gaussian_filter(matrix, sigma=3, mode = 'nearest')
    else :
        matrix = Field[s, 0, :, :]
    plt.figure(figsize=(8,6)) # dpi*figsize=pixel,
    minField = np.min(matrix)
    maxField = np.max(matrix)
    if plotMode == "LPF":
        plt.imshow(matrix, cmap=plt.cm.jet, vmin=minField, vmax=maxField, origin='lower',
               extent=[0, 1000, 0, 1000], interpolation=None, zorder=1)
    else :
        plt.imshow(matrix, cmap=plt.cm.jet, vmin=minField, vmax=maxField, origin='lower',
               extent=[0, 1000, 0, 1000], interpolation='spline16', zorder=1)
    
    #plt.clim(0.9,2)
    # hide the axis label
    plt.xticks([0,250,500,750,1000],fontsize=14)
    plt.yticks([0,250,500,750,1000],fontsize=14)
    # Set the x and y axis labels
    plt.xlabel('x Position / mm', fontsize=22)
    plt.ylabel('z Position / mm', fontsize=21)
    plt.title(ptitle + ' ' + str(sampleNumber), fontsize=22)
    cb = plt.colorbar(cax=None,ax=None,shrink=1.0, format ="%-5.2f",ticks=np.linspace(matrix.min(), matrix.max(), 5))
    if plotMode == "LDF":
        cb.ax.set_title('LDF Level', fontsize=16, pad=10)
    elif plotMode == "LPF":
        cb.ax.set_title('LPF Level', fontsize=16, pad=10)
    else:
        cb.ax.set_title('Heat Flux / $\\mathregular{kW\cdot m^{-2}}$', fontsize=14, pad=10, fontname='Times New Roman')
    cb.ax.tick_params(labelsize=14)
    
    plt.subplots_adjust(left=0.14,right=0.94,bottom=0.1,top=0.9)
    
    plt.savefig(os.path.join(savePath + ptitle + str(sampleNumber) + ".png"), dpi = 200)
    #plt.show()
    del Field
    del matrix
    plt.close()
    #plt.clf()

#----------------------------------------#
