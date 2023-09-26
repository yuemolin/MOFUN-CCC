import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
from utils import verboseprint

# Input data:
#    1. GEP data.txt ("\t"  seperated) (TPM/intensity data)
#           Sample1 Sample2 Sample3
#    Gene1   1       2       3
#    Gene2   4       5       6
#    Gene3   7       8       9

#    2. Methyl data ("\t" seperated) (beta value)
#           Sample1 Sample2 Sample3
#    Probe1   1       2       3
#    Probe2   4       5       6
#    Probe3   7       8       9

#    3. Cell type data
def Count2Frac(x: "np.array")->"np.array":
    '''Convert count data to fraction data
    Input: (row: sample, column: Cell type count)
        x: np.array, count data 
    Output: (row: sample, column: Cell type fraction)
        x: np.array, fraction data normalized by row 
    '''
    x = x/(x.sum(axis = 1,keepdims=True))
    return x

# Count = np.random.rand(4,5)
# Frac = Count2Frac(Count)
# Frac.sum(axis=1)

def getDfRange(df: "pd.DataFrame")->"(float,float)":
    '''Get the range of a dataframe
    Input:
        df: pd.DataFrame
    Output:
        min,max: float, min and max value of the dataframe'''
    return df.min().min(),df.max().max()

# testdf = pd.DataFrame([[1,2,3],[4,5,6]])
# getDfRange(testdf)

def Beta2Mvalue(Beta: "pd.DataFrame",offset:"float"=0.01)->"pd.DataFrame":
    '''Convert beta value to M value
    Input: (row: Probe, column: sample)
        Beta: pd.DataFrame, beta value
        offset: float, offset to avoid log(0) or 1/0
    Output: (row: Probe, column: sample)
        Mvalue: pd.DataFrame, M value
    '''
    assert (Beta.max().max()<=1) & (Beta.min().min()>=0), "Methyl data are not between 0 and 1, input beta value"
    Beta = Beta.where(Beta != 0,offset)
    Beta = Beta.where(Beta != 1, 1-offset)
    return np.log2(Beta/(1-Beta))

# beta = pd.DataFrame(np.random.rand(4,5))
# mvalue =  Beta2Mvalue(beta)

def Mvalue2Beta(Mvalue: "pd.DataFrame")->"pd.DataFrame":
    '''Convert M value to beta value
    Input: (row: Probe, column: sample)
        Mvalue: pd.DataFrame, M value
    Output: (row: Probe, column: sample)
        Beta: pd.DataFrame, beta value'''
    return (1+2**(-1*Mvalue))**(-1)

# beta1 = Mvalue2Beta(mvalue)
# ((beta1 - beta)<1e-5).all().all()

def GEP_cleaning(GEP:"pd.DataFrame",
                 lg2Transform:"bool"=None,
                 mean_cutoff:"float"=0.1,
                 sd_cutoff:"float"=0.1,
                 removeMissing:"bool"=True,
                 verbose:"bool"=True)->"pd.DataFrame":
    '''Clean Gene expression data
    Input: (row: gene, column: sample)
        GEP: pd.DataFrame, GEP data, TPM(RNAseq)/intensity data(microarray)
        mean_cutoff: float, mean cutoff(under raw scale), remove genes with low expression
        sd_cutoff: float, sd cutoff(under raw scale), remove genes with low variance
        removeMissing: bool, whether to remove genes with missing value
        lg2Transform: bool, whether to log2(offset = 1) transform the data, default is None for auto detection
        verbose: bool, 
    Output: (row: gene, column: sample)
        GEP_out: pd.DataFrame, cleaned and transformed GEP data
    '''
    # Count data detection
    assert ~(GEP%1 == 0).all(axis=None), "Count data detected, input TPM/intensity data"
    # log/normal scale detection
    if getDfRange(GEP)[1]<50:
        warnings.warn("Maximum value of GEP data is less than 50, assume data is already in log scale")
        if lg2Transform == None:
            lg2Transform = False
    else:
        if lg2Transform == None:
            lg2Transform = True
    # missing value detection
    
    if GEP.isnull().values.any() and removeMissing:
        GEP = GEP.dropna(axis="index",how="any")
        warnings.warn("Missing value detected, gene were removed")
    inputGene = GEP.shape[0]
    GEP_out = GEP.loc[(GEP.mean(axis=1) >= mean_cutoff) & (np.sqrt(GEP.var(axis=1)) >= sd_cutoff), :]
    outputGene = GEP_out.shape[0]
    verboseprint("Gene data cleaning [%d->%d]: %d genes removed"%(inputGene,outputGene,(inputGene-outputGene)),verbose)

    if lg2Transform:
        GEP_out = np.log2(GEP_out+1)
        verboseprint("Log2 transform applied",verbose)

    return GEP_out

# GEP = pd.read_table("../RworkSpace/Dataset/CountData/EVAPR_rawTPM.txt",index_col=0)

# cleanedGEP = GEP_cleaning(GEP)
# cleanedGEP = GEP_cleaning(GEP,lg2Transform=False)
# cleanedGEP = GEP_cleaning(np.log2(GEP+1),lg2Transform=False)
# missingGEP = GEP.copy()
# missingGEP.iloc[0,0] = np.nan
# cleanedGEP = GEP_cleaning(missingGEP, lg2Transform=False)
# cleanedGEP = GEP_cleaning(GEP,mean_cutoff=0,sd_cutoff=0,verbose=False)
# cleanedGEP = GEP_cleaning(GEP.round())

def Methyl_cleaning(Methyl:"pd.DataFrame",
                    beta_mean_min:"float"=0.1,
                    beta_mean_max:"float"=0.9,
                    to_Mvalue: "bool"=True,
                    removeMissing:"bool"=True,
                    verbose:"bool" = True) -> "pd.DataFrame":
    '''Clean Methylation data
    Input:  (row: Probe, column: sample)
        Methyl: pd.DataFrame, beta value required
        beta_mean_min: float, minimum mean beta value, remove probes with low beta value
        beta_mean_max: float, maximum mean beta value, remove probes with high beta value
        to_Mvalue: bool, whether to convert beta value to M value
        removeMissing: bool, whether to remove probes with missing value
        verbose: bool, 
    Output: (row: Probe, column: sample)
        Methyl_out: pd.DataFrame, cleaned Methyl data
    '''
    v_min,v_max = getDfRange(Methyl)
    assert (v_max <= 1) and (v_min>=0), "Methyl data are not between 0 and 1, input beta value"
    inputProbe = Methyl.shape[0]
    if Methyl.isnull().values.any() and removeMissing:
        Methyl = Methyl.dropna(axis="index",how="any")
        warnings.warn("Missing value detected, probe were removed")
    #Methyl = Methyl.loc[(~ ((Methyl < beta_mean_min).mean(axis=1) > 0.8)) & (~ ((Methyl > beta_mean_max).mean(axis=1) > 0.8)), ]
    Methyl_out = Methyl.loc[(Methyl.mean(axis=1)>beta_mean_min) & (Methyl.mean(axis=1)<beta_mean_max),:]
    outputProbe = Methyl_out.shape[0]
    verboseprint("Methyl data cleaning [%d->%d]: %d probes removed"%((inputProbe,outputProbe,inputProbe-outputProbe)),verbose)

    if to_Mvalue:
        Methyl_out = Beta2Mvalue(Methyl_out)
        verboseprint("Transformed to M value",verbose)
    return Methyl_out

# Methyl = pd.read_table("../RworkSpace/Dataset/CountData/EVAPR_rawBeta.txt",index_col=0)
# cleanedMethyl = Methyl_cleaning(Methyl)
# cleanedMethyl = Methyl_cleaning(Methyl,0,1)
# cleanedMethyl = Methyl_cleaning(Methyl,0,1,False)
# cleanedMethyl = Methyl_cleaning(Methyl,0,1,False,verbose=False)

def transform_Data(Data:"torch.Tensor",
                   transform:"str",
                   axis:"int"=0,
                   verbose:"bool"=True)->"torch.Tensor":
    '''Transform data
    Input:  (row: sample, column: feature)
        Data: torch.Tensor, data to be transformed
        transform: str, "Identity" or "MeanStd" or "Range"
        axis: int, 0 for row, 1 for column (if 1 then the mean and std are 0,1 for each 1 item)
    Output: (row: sample, column: feature)
        Out: torch.Tensor, transformed data
    '''
    assert transform in ["Identity","MeanStd","Range"], "transform method not supported"
    match transform:
        case "Identity":
            Out = Data
        case "MeanStd":
            v_mean= torch.Tensor(np.nanmean(Data, axis=axis, keepdims=True))
            v_std = torch.Tensor(np.nanstd(Data, axis=axis, keepdims=True))
            Out = (Data-v_mean)/(v_std+0.01)
        case "Range":
            v_min = torch.Tensor(np.nanmin(Data, axis=axis, keepdims=True))
            v_max = torch.Tensor(np.nanmax(Data, axis=axis, keepdims=True))
            Out = (Data-v_min)/(v_max-v_min+0.01)
    verboseprint(f"{transform} transformation by axis {axis} is applied", verbose)
    return Out


# Data = torch.randn(10,4)*torch.Tensor([1,20,3,10]) # row feature column sample
# Data.equal(transform_Data(Data,"Identity"))
# transform_Data(Data, "MeanStd",axis=0).mean(0)
# transform_Data(Data, "MeanStd",axis=1).var(1)
# transform_Data(Data, "Range", axis=1).min(1)
# transform_Data(Data, "Range", axis=0).max(1)
# transform_Data(Data, "Range", axis=0).min(1)
# transform_Data(Data, "Range", axis=1).max(1)

def DataAugmentation(GEP:"torch.Tensor",
                     Methylation:"torch.Tensor",
                     CT:"torch.Tensor",
                     NoiseAug = False)->"torch.Tensor":
    '''Data Augmentation
    Input:  (row: sample, column: feature)
        GEP: torch.Tensor, Gene expression data
        Methylation: torch.Tensor, Methylation data
        CT: torch.Tensor, Count data
        NoiseAug: bool, whether to add noise augmentation
    Output: (row: sample, column: feature)
        AugmentedGEP: torch.Tensor, augmented Gene expression data
        AugmentedMethyl: torch.Tensor, augmented Methylation data
        AugmentedCT: torch.Tensor, augmented  Count data
    '''
    GEP_zero = torch.zeros_like(GEP)
    Methyl_zero = torch.zeros_like(Methylation)
    if NoiseAug:
        GEP_noise = torch.rand(GEP.shape)
        Methyl_noise = torch.rand(Methylation.shape)
        AugmentedGEP = torch.cat((GEP,GEP_zero,GEP,GEP_noise,GEP),dim=0)
        AugmentedMethyl = torch.cat((Methyl_zero,Methylation,Methylation,Methylation,Methyl_noise),dim=0)
        AugmentedCT = torch.cat((CT,CT,CT,CT,CT),dim=0)
        return AugmentedGEP,AugmentedMethyl,AugmentedCT
    AugmentedGEP = torch.cat((GEP,GEP_zero,GEP),dim=0)
    AugmentedMethyl = torch.cat((Methyl_zero,Methylation,Methylation),dim=0)
    AugmentedCT = torch.cat((CT,CT,CT),dim=0)
    return AugmentedGEP,AugmentedMethyl,AugmentedCT
# GEP = torch.randn(10,4)
# Methyl = torch.randn(10,4)
# CT = torch.randn(10,4)
# AugmentedGEP,AugmentedMethyl,AugmentedCT = DataAugmentation(GEP,Methyl,CT)

def DataMapping(df:"pd.DataFrame",
                rowID:"list",
                verbose:"bool"=True)->"pd.DataFrame":
    '''Mapping data by provided markers
    Input:  (row: feature, column: sample)
        df: pd.DataFrame, data to be mapped
        rowID: list, markers to be mapped
        verbose: bool,
    Output: (row: sample, column: feature)
        df_out: pd.DataFrame, mapped data
    '''
    commonID = df.index.intersection(rowID)
    missingID = list(set(rowID) - set(commonID))
    commonID_df = df.loc[commonID,]
    missingID_df = pd.DataFrame(
        np.full(shape=(len(missingID), df.shape[1]), fill_value=np.nan))
    missingID_df.columns = df.columns
    missingID_df.index = missingID
    df_out = pd.concat([commonID_df, missingID_df]).loc[rowID]
    verboseprint(f"Out of {len(rowID)} selected markers {len(missingID)} was missing",verbose)
    return df_out
# GEP = pd.read_table("../RworkSpace/Dataset/CountData/EVAPR_rawTPM.txt",index_col=0)
# marker = list(pd.read_table("../RworkSpace/Dataset/EVAPR/GEP_filter_p.txt",index_col=0).index[:10])
# DataMapping(GEP,marker)
# marker.append("test")
# DataMapping(GEP, marker)
# DataMapping(GEP, marker,verbose=False)

class Train_Tri_Data(Dataset):
    def __init__(self, GEP: "pd.DataFrame", Methyl: "pd.DataFrame", CT: "pd.DataFrame",
                 GEP_marker:"list", Methyl_marker:"list",
                 GEP_transform:"str"="Range",Methyl_transform:"str"="Range", by_sample:"bool"=True,
                 Augmentation:"str" = "Zero", scaleCT:"bool"=True,
                 device:"str"='cpu',verbose:"bool"=True):
        '''Train_Tri_Data class
        Input:  
            # Input data
            GEP: pd.DataFrame, Gene expression data (row: Gene, column: sample)
            Methyl: pd.DataFrame, Methylation data (row: Probe, column: sample)
            CT: pd.DataFrame, Count data (row: sample, column: Cell type)

            # Markers
            GEP_marker: list, gene markers to be used
            Methyl_marker: list, methylation markers to be used

            # Data transform
            GEP_transform: str, "Identity" or "MeanStd" or "Range"
            Methyl_transform: str, "Identity" or "MeanStd" or "Range" or "Beta"
            by_sample: bool, transform data by sample, or by feature

            # Auxiliary
            Augmentation: str, "zero" or "noise" or False
            scaleCT: bool, whether to scale count data
            device: str, "cpu" or "cuda"
            verbose: bool,

        Output: 
            GEP: pd.DataFrame, Gene expression data (row: sample, column: gene)
            Methyl: pd.DataFrame, Methylation data (row: sample, column: methylation)
            CT: pd.DataFrame, Count data (row: sample, column: count)
        '''
        self.device = device
        self.GEP_transform = GEP_transform
        self.Methyl_transform = Methyl_transform
        self.normalize_by_sample = by_sample
        self.scalingFactor = torch.Tensor([1.0,1.0,1.0,1.0,1.0]).to(device)
        # GEP data
        # clean GEP data
        verboseprint("========== Cleaning GEP data ==========", verbose)
        GEP0 = GEP_cleaning(GEP, verbose=verbose)
        # take intersection of GEP input and GEP_marker list
        verboseprint("========== Mapping GEP data ==========", verbose)
        self.GEP_feature = list(set(GEP_marker).intersection(set(GEP0.index)))
        verboseprint(f"{len(self.GEP_feature)} are used from {len(GEP_marker)} provided markers", verbose)
        GEP1 = torch.Tensor(GEP0.loc[self.GEP_feature, :].values.T)
        # transform GEP data
        verboseprint("========== Transforming GEP data ==========", verbose)
        axis = 1
        if self.normalize_by_sample:
            axis = 0
        GEP2 = transform_Data(GEP1, GEP_transform,axis=axis, verbose=verbose)
        # Methyl data
        # clean Methyl data
        verboseprint("========== Cleaning DNAm data ==========", verbose)
        Methyl0 = Methyl_cleaning(Methyl, verbose=verbose)
        # take intersection of Methyl input and Methyl_marker list
        verboseprint("========== Mapping DNAm data ==========", verbose)
        self.Methyl_feature = list(set(Methyl_marker).intersection(set(Methyl0.index)))
        verboseprint(f"{len(self.Methyl_feature)} are used from {len(Methyl_marker)} provided markers", verbose)
        Methyl1 = torch.Tensor(Methyl0.loc[self.Methyl_feature, :].values.T)
        # transform Methyl data
        verboseprint("========== Transforming DNAm data ==========", verbose)
        if self.Methyl_transform == "Beta":
            Methyl2 = Mvalue2Beta(Methyl1)
            verboseprint("Beta transformation is applied", verbose)
        else:
            Methyl2 = transform_Data(Methyl1, Methyl_transform,axis=axis,verbose=verbose)
        # CT data
        CT1 = torch.Tensor(CT.loc[:, ("neutrophils", "lymphocytes","monocytes", "eosinophils", "basophils")].values)
        if scaleCT:
            verboseprint("========== Scaling CT data ==========", verbose)
            self.scalingFactor = CT1.median(dim=0)[0]
            CT1 = CT1/self.scalingFactor

        if Augmentation:
            verboseprint("========== Data Augmentation ==========", verbose)
            assert Augmentation in ["Zero", "Noise"], "Augmentation should be 'Zero' or 'Noise'"
            if Augmentation == "Zero":
                NoiseAug = False
            elif Augmentation == "Noise":
                NoiseAug = True
            GEP2, Methyl2, CT1 = DataAugmentation(
                GEP2, Methyl2, CT1, NoiseAug=NoiseAug)
        self.scalingFactor = self.scalingFactor.to(device)
        self.GEP = GEP2.to(device)
        self.Methyl = Methyl2.to(device)
        self.CT = CT1.to(device)
        self.len = self.CT.shape[0]
        self.GEP_Num = self.GEP.shape[1]
        self.Methyl_Num = self.Methyl.shape[1]
        self.CT_Num = self.CT.shape[1]
        verboseprint(f"Output Data:\nGEP: {self.GEP.shape},\nMethyl: {self.Methyl.shape}, \nCT: {self.CT.shape}", verbose)
    def __getitem__(self, index):
        return self.GEP[index, :], self.Methyl[index, :], self.CT[index, :]
    def __len__(self):
        return self.len
    def to(self, device):
        self.device = device
        
        self.GEP = self.GEP.to(device) if self.GEP is not None else None
        self.Methyl = self.Methyl.to(device) if self.Methyl is not None else None
        self.CT = self.CT.to(device) if self.CT is not None else None
        self.scalingFactor = self.scalingFactor.to(device) if self.scalingFactor is not None else None
        return self
# EVAPR_Train_data = Train_Tri_Data(EVAPR_GEP_train, EVAPR_Methyl_train, EVAPR_Count_train,
#                                   GEP_marker=GEP_marker, Methyl_marker=Methyl_marker, device=device, verbose=False)
class Test_Tri_Data(Dataset):
    def __init__(self,
                 Train_Tri_Data: "Train_Tri_Data",
                 GEP:"pd.DataFrame",
                 Methyl:"pd.DataFrame", 
                 true_CT:"pd.DataFrame"=None,
                 device:"str"="cpu",
                 verbose:"bool"=True):
        self.GEP_feature = Train_Tri_Data.GEP_feature
        self.Methyl_feature = Train_Tri_Data.Methyl_feature
        self.GEP_transform = Train_Tri_Data.GEP_transform
        self.Methyl_transform = Train_Tri_Data.Methyl_transform
        self.device = device
        self.normalize_by_sample = Train_Tri_Data.normalize_by_sample
        self.scalingFactor = Train_Tri_Data.scalingFactor.to(device)
        if not (GEP is None or Methyl is None):
            assert GEP.shape[1] == Methyl.shape[1], "GEP and Methyl have different number of samples"
            assert all(GEP.columns == Methyl.columns), "GEP and Methyl have different sample names"
        assert not (GEP is None and Methyl is None) , "Provide at least one modality"
        self.pred_CT = None
        self.prediction_modalities = "Both"
        axis = 1
        if self.normalize_by_sample:
            axis = 0        
        if GEP is not None:
            self.sampleID = GEP.columns
            self.len = GEP.shape[1]
            GEP0 = GEP_cleaning(GEP,mean_cutoff= -99, sd_cutoff=0,removeMissing=False,verbose=verbose)
            verboseprint("========== Mapping GEP data ==========", verbose)
            GEP1 = torch.Tensor(DataMapping(GEP0, self.GEP_feature,verbose=verbose).values.T)
            verboseprint("========== Transforming GEP data ==========", verbose)

            GEP2 = transform_Data(GEP1, self.GEP_transform,axis,verbose).nan_to_num(0)
            GEPmissingNum = GEP2.isnan().sum()
            if GEPmissingNum > 0:
                verboseprint(f"{GEPmissingNum} missing values are replaced by 0",verbose)
            self.GEP = GEP2.nan_to_num(0)

        if Methyl is not None:
            self.sampleID = Methyl.columns
            self.len = Methyl.shape[1]

            Methyl0 = Methyl_cleaning(Methyl,0,1,True,removeMissing=False,verbose=verbose)
            verboseprint("========== Mapping DNAm data ==========", verbose)
            Methyl1 = torch.Tensor(DataMapping(Methyl0, self.Methyl_feature,verbose=verbose).values.T)
            verboseprint("========== Transforming DNAm data ==========", verbose)
            if self.Methyl_transform == "Beta":
                Methyl2 = Mvalue2Beta(Methyl1)
                verboseprint("Beta transformation is applied", verbose)
            else: 
                Methyl2 = transform_Data(Methyl1, self.Methyl_transform,axis,verbose)
            MethylmissingNum = Methyl2.isnan().sum()
            if MethylmissingNum > 0:
                verboseprint(f"{MethylmissingNum} missing values are replaced by 0",verbose)
            self.Methyl = Methyl2.nan_to_num(0)
        self.true_CT = None
        if true_CT is not None:
            self.true_CT = torch.tensor(true_CT.loc[:, ("neutrophils", "lymphocytes", "monocytes",
                                        "eosinophils", "basophils")].values, dtype=torch.float)
        try:
            GEP2
        except NameError:
            self.GEP = torch.zeros(self.len, len(self.GEP_feature))
            self.prediction_modalities = "Methyl"
        try:
            Methyl2
        except NameError:
            self.Methyl = torch.zeros(self.len, len(self.Methyl_feature))
            self.prediction_modalities = "GEP"
        self.GEP = self.GEP.to(device)
        self.Methyl = self.Methyl.to(device)
        if not self.true_CT is None:
            self.true_CT = self.true_CT.to(device)
        verboseprint(
            f"Testing Data:\nGEP: {self.GEP.shape},\nMethyl: {self.Methyl.shape}", verbose)

    def __getitem__(self, index):
        if not self.true_CT is None:
            return self.GEP[index, :], self.Methyl[index, :], self.true_CT[index, :]
        else:
            return self.GEP[index, :], self.Methyl[index, :]
         
    def __len__(self):
        return self.len
    
    def predict_CT(self, model, modalities = None):
        '''Predict the cell type composition of the test data
        Args:
            model: the trained model
            modalities: the modalities used for prediction, should be in "Both", "GEP", "Methyl"
        Returns:
            pred_CT: the predicted cell type composition
        '''
        model.eval()
        model.to(self.device)
        assert modalities in [
            "Both", "GEP", "Methyl", None], "modalities should be in 'Both', 'GEP', 'Methyl'"
        if modalities is None:
            modalities = self.prediction_modalities
        if self.prediction_modalities != "Both":
            if modalities != self.prediction_modalities:
                print("Only", self.prediction_modalities, "is available, the prediction modalities is set to be ", self.prediction_modalities)
            modalities = self.prediction_modalities

        if modalities == "GEP":
            Methyl0 = torch.zeros(self.len, len(self.Methyl_feature)).to(self.device)
            pred_CT = model.forward(self.GEP, Methyl0)
        elif modalities == "Methyl":
            GEP0 = torch.zeros(self.len, len(self.GEP_feature)).to(self.device)
            pred_CT = model.forward(GEP0, self.Methyl)
        else:
            pred_CT = model.forward(self.GEP, self.Methyl)
        print("Cell counts are predicted by ", modalities, " modalities")
        pred_CT[pred_CT < 0] = 0
        self.pred_CT = pred_CT*self.scalingFactor
        return self.pred_CT
    
    def save_pred_CT(self, filename):
        '''Save the predicted cell type composition to a csv file
        Args:
            filename: the filename of the csv file
        '''
        if self.pred_CT is None:
            print("No prediction is available")
        else:
            pred_CT = pd.DataFrame(self.pred_CT.cpu().detach().numpy(), 
                                   columns=["neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils"],
                                   index = self.sampleID)
            pred_CT.to_csv(filename)
            print("The predicted cell type composition is saved to ", filename)
# EVAPR_Test_data = Test_Tri_Data(
#     EVAPR_Train_data, EVAPR_GEP_test, EVAPR_Methyl_test, EVAPR_Count_test, device=device)
