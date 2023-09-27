from torch import nn as nn
import torch
from torchmetrics import SpearmanCorrCoef,PearsonCorrCoef
import pickle
import sys
import time
from threading import Thread
import warnings

Logo = '''
███╗   ███╗ ██████╗ ███████╗██╗   ██╗███╗   ██╗       ██████╗ ██████╗ ██████╗
████╗ ████║██╔═══██╗██╔════╝██║   ██║████╗  ██║      ██╔════╝██╔════╝██╔════╝
██╔████╔██║██║   ██║█████╗  ██║   ██║██╔██╗ ██║█████╗██║     ██║     ██║
██║╚██╔╝██║██║   ██║██╔══╝  ██║   ██║██║╚██╗██║╚════╝██║     ██║     ██║
██║ ╚═╝ ██║╚██████╔╝██║     ╚██████╔╝██║ ╚████║      ╚██████╗╚██████╗╚██████╗
╚═╝     ╚═╝ ╚═════╝ ╚═╝      ╚═════╝ ╚═╝  ╚═══╝       ╚═════╝ ╚═════╝ ╚═════╝
'''

def verboseprint(msg, verbose):
    if verbose:
        print(msg)

def pickle_save(obj,filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    # print("Data saved")

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def Mat_MSE(x,y,reduction = False):
    MSE_fn = nn.MSELoss(reduction="none")
    MSE_mat = MSE_fn(x,y).mean(dim = 0)
    if reduction:
        MSE_mat = MSE_mat.mean(dim = 0,keepdims = True)
    return MSE_mat

#x = torch.Tensor([[1,2,3],[4,5,6]])
#y = torch.Tensor([[1.1,2.2,3.3],[4.4,5.5,6.6]])
#Mat_MSE(x,y)
#Mat_MSE(x,y,reduction = True)

def Mat_spearman_corr(x,y,reduction = False):
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        spearman = SpearmanCorrCoef(num_outputs=x.shape[1])
    # spearman(x,y)
    # out = []
        if reduction:
            spearman = SpearmanCorrCoef()
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
    # for i in range(y.shape[1]):
    #     out.append(stats.spearmanr(x[...,i], y[...,i])[0])
    # # return torch.Tensor(out)
    return spearman(x,y)

def Mat_pearson_corr(x,y,reduction = False):
    pearson = PearsonCorrCoef(num_outputs=x.shape[1])

    if reduction:
        pearson = PearsonCorrCoef()
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
    return pearson(x,y)

def Mat_ccc(x,y,reduction = False):
    ''' Concordance Correlation Coefficient'''
    if reduction:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
    out = []
    for i in range(y.shape[1]):
        x_i = x[...,i]
        y_i = y[...,i]
        # sxy = ((x_i - x_i.mean())*(y_i - y_i.mean())).sum()/x_i.shape[0]
        # ccc = 2*sxy / (np.var(x_i) + np.var(y_i) + (x_i.mean() - y_i.mean())**2)
        ccc = 2*(torch.cov(torch.stack((x_i,y_i)))[1,0]) / (x_i.var() + y_i.var() + (x_i.mean() - y_i.mean())**2)
        out.append(ccc)
    return torch.Tensor(out)

#data = torch.Tensor([[127, 134], [118, 120], [135, 142], [123, 115], [112, 117], [119, 123], [125, 125], [109, 111], [106, 122], [110, 115], [111, 109], [125, 127], [119, 118], [140, 139], [115, 111], [110, 110], [122, 122], [119, 115], [134, 137], [105, 105], [117, 112], [111, 113], [114, 115], [126, 119], [113, 115], [129, 127], [130, 129], [130, 130], [115, 120], [131, 126]])
#Mat_ccc(data[:,0],data[:,1],reduction = True) - 0.8649 < 1e-4

def generate_custom_message(message):
    border = "=" * 80

    # Calculate the number of spaces needed on each side to center the message
    total_spaces = 78 - len(message)  # 78 spaces account for the '#' characters and spaces on both sides
    left_spaces = total_spaces // 2
    right_spaces = total_spaces - left_spaces

    formatted_message = f"{border}\n#{' ' * left_spaces}{message}{' ' * right_spaces}#\n{border}\n"
    print(formatted_message)

class Spinner:
    busy = False
    delay = 0.1

    def __init__(self, delay=None):
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            for char in "|/-\\":
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(self.delay)
                sys.stdout.write('\b')
                sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False