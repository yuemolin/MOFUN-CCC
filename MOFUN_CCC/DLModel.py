import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchmetrics.functional import pairwise_cosine_similarity
from utils import Mat_MSE, Mat_spearman_corr, Mat_ccc, pickle_save, pickle_load, verboseprint
from sklearn.linear_model import LassoCV
import copy

import time

class BaseModule(nn.Module):
    def __init__(self, loss, activation):
        super(BaseModule, self).__init__()
        self.getLoss(loss)
        self.getActivation(activation)

    def forward(self):
        NotImplementedError

    def getLoss(self, loss):
        assert loss in ['L2loss', 'L1loss','CrossEntropy'], "Loss not supported"

        class CrossEntropyLoss(nn.Module):
            def forward(self, input, target):
                return nn.functional.cross_entropy(torch.log(input), target)

        Losses = nn.ModuleDict([
            ['L2loss', nn.MSELoss()],
            ['L1loss', nn.L1Loss()],
            ['CrossEntropy', CrossEntropyLoss()],
        ])
        self.loss_fn = Losses[loss]

    def basic_block(self, model_dim, activation, bias=True):
        block = nn.Sequential()
        for i in range(len(model_dim)-1):
            block.add_module(f'Linear_{i}', nn.Linear(
                model_dim[i], model_dim[i+1], bias=bias))
            if i != (len(model_dim)-2):
                block.add_module(f'Activation_{i}', activation)
        return block
    
    def getActivation(self, activation):
        assert activation in ['relu','leakyRelu','Elu','Celu','Gelu']
        activations = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['leakyRelu',nn.LeakyReLU()],
            ["Elu",nn.ELU()],
            ["Celu",nn.CELU()],
            ["Gelu",nn.GELU()],
        ])
        self.activation_fn = activations[activation]

    def getOptimizer(self, lr, momentum, optimizer="SGD"):
        assert optimizer in ['SGD', 'Adam']
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, momentum=momentum)
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def train_1epoch(self, train_dataLoader):
        for *X, y in train_dataLoader:
            self.optimizer.zero_grad()
            pred = self.forward(*X)
            loss = self.loss_fn(pred,y) 
            loss.backward()
            self.optimizer.step()

    def metric_eval(self, pred, true, metrics, reduction=False):
        assert [metrics in ["Loss", "MSE", "CCC", "Rsp"]], "metrics should be one of these: Loss,MSE,CCC,Rsp"
        match metrics:
            case "Loss":
                return self.loss_fn(pred, true)
            case "MSE":
                return Mat_MSE(pred, true, reduction)
            case "CCC":
                return Mat_ccc(pred, true, reduction)
            case "Rsp":
                return Mat_spearman_corr(pred, true, reduction)

    def test_1epoch(self, train_oneLoader, test_oneLoader=False, reduction=False):
        assert hasattr(self, 'loss_fn'), "No loss function"
        *Train_X, Train_y = next(iter(train_oneLoader))
        if test_oneLoader:
            *Test_X, Test_y = next(iter(test_oneLoader))
        with torch.no_grad():
            Train_pred = (self.forward(*Train_X)).cpu()
            Train_true = Train_y.cpu()
            TrainLoss = self.metric_eval(
                Train_pred, Train_true, metrics="Loss", reduction=reduction)
            TrainMSE = self.metric_eval(
                Train_pred, Train_true, metrics="MSE", reduction=reduction)
            TrainRsp = self.metric_eval(
                Train_pred, Train_true, metrics="Rsp", reduction=reduction)
            TrainCCC = self.metric_eval(
                Train_pred, Train_true, metrics="CCC", reduction=reduction)
            if test_oneLoader:
                Test_pred = (self.forward(*Test_X)).cpu()
                Test_true = Test_y.cpu()
                TestLoss = self.metric_eval(
                    Test_pred, Test_true, metrics="Loss", reduction=reduction)
                TestMSE = self.metric_eval(
                    Test_pred, Test_true, metrics="MSE", reduction=reduction)
                TestRsp = self.metric_eval(
                    Test_pred, Test_true, metrics="Rsp", reduction=reduction)
                TestCCC = self.metric_eval(
                    Test_pred, Test_true, metrics="CCC", reduction=reduction)
                return torch.stack((TrainLoss, TestLoss)), torch.stack((TrainMSE,TestMSE)), torch.stack((TrainRsp, TestRsp)), torch.stack((TrainCCC, TestCCC))
            return TrainLoss.reshape(1), TrainMSE.reshape(1,-1), TrainRsp.reshape(1,-1), TrainCCC.reshape(1,-1)

    def runExp(self, train_data, lr, batch_size, epochs, optimizer="SGD", momentum=0, test_data=False, reduction=False):
        self.device = next(self.parameters()).device
        print(f"working on {self.device}")
        self.getOptimizer(lr, momentum=momentum, optimizer=optimizer)
        train_dataLoader = DataLoader(
            dataset=train_data, batch_size=int(batch_size), shuffle=True)
        train_oneLoader = DataLoader(
            dataset=train_data, batch_size=len(train_data), shuffle=False)
        test_oneLoader = False
        if test_data:
            test_oneLoader = DataLoader(
                dataset=test_data, batch_size=len(test_data), shuffle=False)
        Loss_value = []
        MSE_value = []
        corrSp_value = []
        ccc_value = []
        totaltime = []
        for _ in tqdm(range(epochs)):
            # self.train()
            cpu = self.device.type == "cpu"
            if cpu:
                start_time = time.time()
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            self.train_1epoch(train_dataLoader)
            if cpu:
                totaltime.append(time.time() - start_time)
            else:
                end.record()
                torch.cuda.synchronize()
                totaltime.append(start.elapsed_time(end))
            self.eval()
            Loss, MSE, Rsp, CCC = self.test_1epoch(
                train_oneLoader, test_oneLoader=test_oneLoader, reduction=reduction)
            Loss_value.append(Loss), MSE_value.append(
                MSE), corrSp_value.append(Rsp),  ccc_value.append(CCC)
        Loss_value = torch.stack(Loss_value).permute(1, 0)
        MSE_value = torch.stack(MSE_value).permute(1, 0, 2)
        corrSp_value = torch.stack(corrSp_value).permute(1, 0, 2)
        ccc_value = torch.stack(ccc_value).permute(1, 0, 2)

        # Loss_value: [0/1:Train/Test, #Epoches]
        # MSE_value/corr_value:  [0/1:Train/Test, #Epoches, 5CTs]
        Stats_dict = {
            "Loss": Loss_value,
            "MSE": MSE_value,
            "corrSp": corrSp_value,
            "CCC": ccc_value,
            "EpochTrainTime": f"{np.array(totaltime[int(epochs*0.1):]).mean()} ms on {self.device}"
        }
        return Stats_dict

    def save_model(self, filePath):
        torch.save(self.state_dict(), filePath)
        print("Model saved")

    def load_model(self, filePath, device='cpu'):
        self.load_state_dict(torch.load(
            filePath, map_location=torch.device(device)))
        self.eval()

class SingleModalNN(BaseModule):

    def __init__(self, layer_dim, loss, activation, dropout1st=0):
        """
        layer_dim = (16221,500,50,5)
        """
        super(SingleModalNN, self).__init__(loss, activation)

        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim)
             for in_dim, out_dim in zip(layer_dim, layer_dim[1:])]
        )
        self.dropout = nn.Dropout(dropout1st)

    def forward(self, x, y):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                x = self.dropout(self.activation_fn(layer(x)))
            elif idx == len(self.layers)-1:
                out = layer(x)
            else:
                x = self.activation_fn(layer(x))
        return out

class Residual_Block(nn.Module):
    def __init__(self, Layer_dim, activation_fn):
        super().__init__()
        self.fusion = self.basic_block(Layer_dim, activation_fn)
        self.layer_norm_GEP = nn.BatchNorm1d(Layer_dim[-1])
        self.layer_norm_Methyl = nn.BatchNorm1d(Layer_dim[-1])

    def basic_block(self, model_dim, activation):
        block = nn.Sequential()
        for i in range(len(model_dim)-1):
            block.add_module(f'Linear_{i}', nn.Linear(
                model_dim[i], model_dim[i+1]))
            if i != (len(model_dim)-2):
                block.add_module(
                    f'BatchNorm_{i}', nn.BatchNorm1d(model_dim[i+1]))
                block.add_module(f'Activation_{i}', activation)
        return block

    def forward(self, GEP, Methyl):
        Fused_Feature = self.fusion(torch.concat((GEP, Methyl), dim=1))
        ReLu = torch.nn.ReLU()
        GEP_normalized = ReLu(GEP + self.layer_norm_GEP(Fused_Feature))
        Methyl_normalized = ReLu(
            Methyl + self.layer_norm_Methyl(Fused_Feature))
        return GEP_normalized, Methyl_normalized

class LassoCVModel():
    def __init__(self, ):
        self.TrainFlag = False

    def Train(self, Xmat, Ymat):
        self.CV_Models = []
        for CT_index in range(5):
            X, y = Xmat, Ymat[:, CT_index]
            model = LassoCV(max_iter = 300, n_jobs = 10,random_state=0, n_alphas=10)
            model.fit(X, y)
            self.CV_Models.append(model)
            print(CT_index, " is Done ------------------")
        self.TrainFlag = True

    def Predict(self, test_Xmat):
        assert self.TrainFlag == True, "Model has not been trained yet!"
        Res = []
        for model in self.CV_Models:
            yhat = model.predict(test_Xmat)
            Res.append(yhat)
        Pred_Res = (torch.Tensor(np.array(Res)).T)
        return Pred_Res

class MOFUN_CCC(BaseModule):

    def __init__(self, layer_dim, loss, activation, dropout1st=0):

        super(MOFUN_CCC, self).__init__(loss, activation)
        self.model_structure = layer_dim
        self.Dropout = nn.Dropout(dropout1st)
        self.GEPextract = self.basic_block(
            layer_dim["GEP_extract"], self.activation_fn)
        self.Methylextract = self.basic_block(
            layer_dim["Methyl_extract"], self.activation_fn)
        self.ResStructure = nn.ModuleList([])
        for _ in range(layer_dim["n_Res"]):
            ResObj = Residual_Block(layer_dim["fusion"], self.activation_fn)
            self.ResStructure.append(ResObj)
        self.Output = self.basic_block(layer_dim["Out"], self.activation_fn)

    def outputLayer(self, GEP, Methyl):
        Combined_feature = torch.concat((GEP, Methyl), dim=1)
        Fused_output = self.Output(Combined_feature)
        return Fused_output

    def getWeight(self, GEPraw, GEPfused, Methylraw, Methylfused):
        Methyl_score = torch.diagonal(pairwise_cosine_similarity(
            Methylraw, Methylfused)).unsqueeze(1)
        GEP_score = torch.diagonal(
            pairwise_cosine_similarity(GEPraw, GEPfused)).unsqueeze(1)
        scoreMat = torch.concat((GEP_score, Methyl_score), dim=1)
        return scoreMat

    def forward(self, GEP, Methyl):

        GEP_feature = self.GEPextract(self.Dropout(GEP))
        Methyl_feature = self.Methylextract(self.Dropout(Methyl))
        GEP_rawfeature = GEP_feature.detach().clone()
        Methyl_rawfeature = Methyl_feature.detach().clone()
        for ResObj in self.ResStructure:
            GEP_feature, Methyl_feature = ResObj(GEP_feature, Methyl_feature)
        self.weightMat = self.getWeight(
            GEP_rawfeature, GEP_feature, Methyl_rawfeature, Methyl_feature)
        Fused_output = self.Output(torch.concat(
            (GEP_feature, Methyl_feature), dim=1))
        return Fused_output

def Model_save(Model, Training_Data,folderPath):
    # a higher level saving function wraper to save the model and training data 
    # so that the model can be directly called and used later
    Train_shell = copy.deepcopy(Training_Data)
    Train_shell.GEP = Train_shell.Methyl = Train_shell.Count = None
    Train_shell = Train_shell.to("cpu")
    pickle_save(Train_shell, f"{folderPath}/TrainShell.pkl")
    pickle_save(Model.model_structure, f"{folderPath}/Model_str.pkl")
    Model.save_model(f"{folderPath}/Model_dict.pt")

def Model_load(folderPath, device = "cpu"):
    # a higher level loading function wraper to load the model and training data
    Model_structure = pickle_load(f"{folderPath}/Model_str.pkl")
    Training_Shell = pickle_load(f"{folderPath}/TrainShell.pkl")
    Model = MOFUN_CCC(Model_structure, loss="L1loss",activation="relu", dropout1st=0)
    Model.load_model(f"{folderPath}/Model_dict.pt",device = device)
    return Training_Shell, Model
