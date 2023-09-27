from scipy import stats
from torch import nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import pickle
from utils import *
from Data_prep import getDfRange

# TODO: count prediction 的负的给去掉
# 图改自适应 range上下加一些
def deconRes_plot(Loss_value, MSE_value, corr_value, save=False):

    fig = figure(figsize=(20, 18), dpi=100)
    plt.subplot(3, 2, 1)  # row 1, col 2 index 1

    plt.plot(Loss_value[0], label="Loss Function")
    plt.legend(loc='lower right')
    plt.title("Loss of training set")
    plt.xlabel("number of epoch")
    plt.ylabel("Loss")

    plt.subplot(3, 2, 2)  # row 1, col 2 index 1

    plt.plot(Loss_value[1], label="Loss Function")
    plt.legend(loc='lower right')
    plt.title("Loss of Testing set")
    plt.xlabel("number of epoch")
    plt.ylabel("Loss")

    labels = ["neutrophils", "lymphocytes",
              "monocytes", "eosinophils", "basophils"]
    plt.subplot(3, 2, 3)  # row 1, col 2 index 1
    for i in range(MSE_value.shape[2]):
        plt.plot(MSE_value[0, :, i], label=labels[i])
    # plt.legend(loc='lower right')
    plt.title("MSE of training set")
    plt.xlabel("number of epoch")
    plt.ylabel("MSE")

    plt.subplot(3, 2, 4)  # index 2
    for i in range(MSE_value.shape[2]):
        plt.plot(MSE_value[1, :, i], label=labels[i])
    plt.legend(loc='lower right')
    plt.title("MSE of test set")
    plt.xlabel("number of epoch")
    plt.ylabel("MSE")

    plt.subplot(3, 2, 5)  # row 1, col 2 index 1
    for i in range(corr_value.shape[2]):
        plt.plot(corr_value[0, :, i], label=labels[i])
    # plt.legend(loc='lower right')
    plt.title("Spearman correlation of training set")
    plt.xlabel("number of epoch")
    plt.ylabel("R")

    plt.subplot(3, 2, 6)  # index 2
    for i in range(corr_value.shape[2]):
        plt.plot(corr_value[1, :, i], label=labels[i])
    plt.legend(loc='lower right')
    plt.title("Spearman correlation of test set")
    plt.xlabel("number of epoch")
    plt.ylabel("R")
    if save:
        plt.savefig(save)
        plt.close(fig)
    else:
        plt.show()

def indCT_scatter_plt(pred,true,save=False):
    plot_labs = ["neutrophils","lymphocytes","monocytes","eosinophils","basophils"]
    with torch.no_grad():
        r = Mat_spearman_corr(pred,true)
        mse = Mat_MSE(pred,true)
        # torch.stack((torch.concat((pred,true)).max(axis=0).values,torch.concat((pred,true)).min(axis=0).values),dim=0)
        figure(figsize=(30, 5), dpi=100)
        plt.subplot(1, 5, 1) # row 1, col 2 index 1
        plt.scatter(pred[:,0],true[:,0],label = plot_labs[0] )
        plt.axline((0, 0), slope=1,color = "grey",linestyle="--")
        plt.text(0.5, 0.2, 'r = {:.4}'.format(r[0]), fontsize = 15,c = "red")
        plt.text(0.5, 0.1, 'MSE = {:.2E}'.format(mse[0]), fontsize = 15,c = "red")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("Predicted prop")
        plt.ylabel("True prop")
        plt.title("Neutrophils Scatter plot")

        plt.subplot(1, 5, 2)
        plt.scatter(pred[:,1],true[:,1],label = plot_labs[0] )
        plt.axline((0, 0), slope=1,color = "grey",linestyle="--")
        plt.xlim(0,0.8)
        plt.ylim(0,0.8)
        plt.text(0.4, 0.16, 'r = {:.4}'.format(r[1]), fontsize = 15,c = "red")
        plt.text(0.4, 0.1, 'MSE = {:.2E}'.format(mse[1]), fontsize = 15,c = "red")
        plt.xlabel("Predicted prop")
        plt.ylabel("True prop")
        plt.title("Lymphocytes Scatter plot")

        plt.subplot(1, 5, 3) 
        plt.scatter(pred[:,2],true[:,2],label = plot_labs[0] )
        plt.axline((0, 0), slope=1,color = "grey",linestyle="--")
        plt.xlim(0,0.2)
        plt.ylim(0,0.2)
        plt.xlabel("Predicted prop")
        plt.ylabel("True prop")
        plt.text(0.1, 0.03, 'r = {:.4}'.format(r[2]), fontsize = 15,c = "red")
        plt.text(0.1, 0.01, 'MSE = {:.2E}'.format(mse[2]), fontsize = 15,c = "red")
        plt.title("Monocytes Scatter plot")

        plt.subplot(1, 5, 4) 
        plt.scatter(pred[:,3],true[:,3],label = plot_labs[0] )
        plt.axline((0, 0), slope=1,color = "grey",linestyle="--")
        plt.xlim(0,0.3)
        plt.ylim(0,0.3)
        plt.text(0.15, 0.06, 'r = {:.4}'.format(r[3]), fontsize = 15,c = "red")
        plt.text(0.15, 0.03, 'MSE = {:.2E}'.format(mse[3]), fontsize = 15,c = "red")
        plt.xlabel("Predicted prop")
        plt.ylabel("True prop")
        plt.title("Eosinophils Scatter plot")

        plt.subplot(1, 5, 5) 
        plt.scatter(pred[:,4]*100,true[:,4]*100,label = plot_labs[0] )
        plt.axline((0, 0), slope=1,color = "grey",linestyle="--")
        plt.text(1.25, 0.4, 'r = {:.4}'.format(r[4]), fontsize = 15,c = "red")
        plt.text(1.25, 0.2, 'MSE = {:.2E}'.format(mse[4]), fontsize = 15,c = "red")
        plt.xlim(0,2)
        plt.ylim(0,2)
        plt.xlabel("Predicted prop%")
        plt.ylabel("True prop%")
        plt.title("Basophils Scatter plot")
        if save:
            plt.savefig(save)
        else: 
            plt.show()

def Count_scatter_plt(pred, true, save=False):
    plot_labs = ["neutrophils", "lymphocytes",
                 "monocytes", "eosinophils", "basophils"]
    with torch.no_grad():
        r = Mat_spearman_corr(pred, true)
        mse = Mat_MSE(pred, true)

        plt.figure(figsize=(30, 5), dpi=100)
        plt.subplot(1, 5, 1)  # row 1, col 2 index 1
        plt.scatter(pred[:, 0], true[:, 0], label=plot_labs[0])
        plt.axline((0, 0), slope=1, color="grey", linestyle="--")
        plt.text(6, 2, 'r = {:.4}'.format(r[0]), fontsize=15, c="red")
        plt.text(6, 1, 'MSE = {:.2E}'.format(mse[0]), fontsize=15, c="red")
        plt.xlim(0, 12)
        plt.ylim(0, 12)
        plt.xlabel("Predicted Count")
        plt.ylabel("True Count")
        plt.title("Neutrophils Scatter plot")

        plt.subplot(1, 5, 2)
        plt.scatter(pred[:, 1], true[:, 1], label=plot_labs[0])
        plt.axline((0, 0), slope=1, color="grey", linestyle="--")
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.text(2.5, 1, 'r = {:.4}'.format(r[1]), fontsize=15, c="red")
        plt.text(2.5, 0.7, 'MSE = {:.2E}'.format(mse[1]), fontsize=15, c="red")
        plt.xlabel("Predicted Count")
        plt.ylabel("True Count")
        plt.title("Lymphocytes Scatter plot")

        plt.subplot(1, 5, 3)
        plt.scatter(pred[:, 2], true[:, 2], label=plot_labs[0])
        plt.axline((0, 0), slope=1, color="grey", linestyle="--")
        plt.xlim(0, 1.2)
        plt.ylim(0, 1.2)
        plt.xlabel("Predicted Count")
        plt.ylabel("True Count")
        plt.text(0.8, 0.5, 'r = {:.4}'.format(r[2]), fontsize=15, c="red")
        plt.text(0.8, 0.3, 'MSE = {:.2E}'.format(
            mse[2]), fontsize=15, c="red")
        plt.title("Monocytes Scatter plot")

        plt.subplot(1, 5, 4)
        plt.scatter(pred[:, 3], true[:, 3], label=plot_labs[0])
        plt.axline((0, 0), slope=1, color="grey", linestyle="--")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.text(0.6, 0.2, 'r = {:.4}'.format(r[3]), fontsize=15, c="red")
        plt.text(0.6, 0.15, 'MSE = {:.2E}'.format(
            mse[3]), fontsize=15, c="red")
        plt.xlabel("Predicted Count")
        plt.ylabel("True Count")
        plt.title("Eosinophils Scatter plot")

        plt.subplot(1, 5, 5)
        plt.scatter(pred[:, 4], true[:, 4], label=plot_labs[0])
        plt.axline((0, 0), slope=1, color="grey", linestyle="--")
        plt.text(0.06, 0.025, 'r = {:.4}'.format(r[4]), fontsize=15, c="red")
        plt.text(0.06, 0.01, 'MSE = {:.2E}'.format(
            mse[4]), fontsize=15, c="red")
        plt.xlim(0, 0.12)
        plt.ylim(0, 0.12)
        plt.xlabel("Predicted Count")
        plt.ylabel("True Count")
        plt.title("Basophils Scatter plot")
        if save:
            plt.savefig(save)
        else:
            plt.show()

def Evaluation_helper(Pred, Meassured,filePath=False):
    EvalMat = pd.DataFrame(torch.stack(
        (Mat_ccc(Pred, Meassured), 
         torch.sqrt(Mat_MSE(Pred, Meassured)), 
         Mat_spearman_corr(Pred, Meassured)),
        dim=0).detach().numpy())
    EvalMat.index = ["CCC","RMSE","Rsp"]
    if filePath:
        #if getDfRange(Meassured)[1]>1:
        #    Count_scatter_plt(Pred, Meassured, filePath+"EvalPlot.png")
        #else :
        #    indCT_scatter_plt(Pred, Meassured, filePath+"EvalPlot.png")
        EvalMat.to_csv(filePath+"EvalMat.csv")
    return EvalMat