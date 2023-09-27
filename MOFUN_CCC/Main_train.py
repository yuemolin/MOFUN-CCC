import argparse
import pandas as pd
from utils import Logo
from Data_prep import GEP_cleaning, Methyl_cleaning
from Feature_selection import MarginalRegression,getTopK_commonfeatures
import os
import torch
from Data_prep import Train_Tri_Data
import json
from DLModel import MOFUN_CCC
from DLModel import Model_save
from utils import pickle_save,generate_custom_message,Spinner

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def get_args_parser():
    parser = argparse.ArgumentParser('MOFUN-CCC model training', add_help=False)
    # Input/Output parameters
    parser.add_argument('--Count', type=str, required=True,
                        help='tab seperated txt cell counts file (rows: Samples, columns: Cell types)')
    parser.add_argument('--RNA', type=str, required=True,
                        help='tab seperated txt gene TPM file (rows: Genes, columns: Samples)')
    parser.add_argument('--DNAm', type=str,required=True,
                        help='tab seperated txt CpG Beta file (rows: CpGs, columns: Samples)')
    parser.add_argument('--Output', type=str,required=True,
                        help='Output folder path for model saving')
    #===============================================================================
    # Marker parameters
    parser.add_argument('--GEP_Marker', type=str,
                        help='Path of the marker file, list of Gene names or Association matrix, generate when leaving blank')
    parser.add_argument('--DNAm_Marker', type=str,
                        help='Path of the marker file, list of CpG names or Association matrix, generate when leaving blank')
    parser.add_argument('--Marker_Method', type=str, choices=["FC", "P"],
                        help='Marker selection method, FC: by fold change, P: by p-value')
    parser.add_argument('--RNA_Marker_num', type=int, default=6000,
                        help="Number of RNA markers to select based on Marker_Method")
    parser.add_argument('--DNAm_Marker_num', type=int, default=6000,
                        help="Number of DNAm markers to select based on Marker_Method")
    #===============================================================================
    # Training Data operation parameters
    parser.add_argument('--RNA_transform', type=str,default= "Range", choices=["Identity", "MeanStd", "Range"],
                        help='RNA transformation method')
    parser.add_argument('--DNAm_transform', type=str,default= "Range", choices=["Identity", "MeanStd", "Range", "Beta"],
                        help='DNAm transformation method')
    parser.add_argument('--transform_by_feature', action=argparse.BooleanOptionalAction,
                        help='Transform by feature if claimed otherwise by sample')
    parser.add_argument('--Data_augmentation', type=str, default="Zero", choices=["Zero", "Noise", "No"],
                        help='Data augmentation method')
    parser.add_argument('--scale_cellcounts', action=argparse.BooleanOptionalAction,default=True,
                        help='Scale cell counts by total cell counts')
    #===============================================================================
    # Model parameters
    parser.add_argument('--Model', type=str,default= "./Models/Default_structure.json",
                        help='Model structure file path (Json format)')
    parser.add_argument('--Loss', type=str,default= "L1loss", choices=["L1loss", "L2loss", "CrossEntropy"],
                        help='Loss function')
    parser.add_argument("--Activation", type=str, default="relu", choices=["relu", "leakyRelu", "Elu", "Celu", "Gelu"],
                        help="Activation function")
    parser.add_argument("--Dropout", default=0.2,type=restricted_float,
                        help="Dropout rate of the first layer")
    parser.add_argument("--Learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--Batch_num", type=int, default=80,
                        help="the number of Batches")
    parser.add_argument("--Epochs", type=int, default=100,
                        help="the number of Epochs")
    #===============================================================================
    # Other parameters
    parser.add_argument('--device', type=str, default="detect", choices=["cuda", "cpu", "detect"],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=424,
                        help='Random seed')
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Show this help message and exit')

    #===============================================================================
    return parser


def main(args):
    Count_Data = pd.read_table(args.Count, index_col=0)
    GEP_Data = pd.read_table(args.RNA, index_col=0)
    DNAm_Data = pd.read_table(args.DNAm, index_col=0)
    OutputFolder = args.Output
    if not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder)
    if not os.path.isdir(OutputFolder):
        raise ValueError("Output path is not a folder")
    generate_custom_message("Data loaded Successfully")
    #===============================================================================
    # Marker selection
    RNA_Marker_num = args.RNA_Marker_num
    DNAm_Marker_num = args.DNAm_Marker_num
    # RNA Marker
    if args.GEP_Marker==None:
        # Create marker file for user
        generate_custom_message("GEP Marker file was not input, generating")
        GEP_cleaned = GEP_cleaning(GEP_Data) 
        Marker_Method = "FC" if args.Marker_Method == None else args.Marker_Method
        with Spinner():
            MarginalRegression(Count_Data,GEP_cleaned,filePath=f"{OutputFolder}/Gene_Marker_{Marker_Method}.txt", by = Marker_Method)
        GEP_marker_File = pd.read_table(f"{OutputFolder}/Gene_Marker_{Marker_Method}.txt", index_col=0)
        GEP_marker = getTopK_commonfeatures(RNA_Marker_num, GEP_marker_File, by_pvalue=Marker_Method=="P")
    else:
        GEP_marker_File = pd.read_table(args.GEP_Marker, index_col=0)
        if GEP_marker_File.shape[1] == 0:
            print("GEP Marker list was input")
            GEP_marker = GEP_marker_File.index.tolist()
            if len(GEP_marker) > RNA_Marker_num:
                GEP_marker = GEP_marker[:RNA_Marker_num]
        else:
            print("GEP Marker association matrix was input")
            if args.Marker_Method==None:
                raise ValueError("Please specify Marker_Method")
            GEP_marker = getTopK_commonfeatures(RNA_Marker_num, GEP_marker_File, by_pvalue=args.Marker_Method=="P")
    # DNAm Marker
    if args.DNAm_Marker==None:
        # Create marker file for user
        generate_custom_message("DNAm Marker file was not input, generating")
        print("This may take a while")
        DNAm_cleaned = Methyl_cleaning(DNAm_Data) 
        Marker_Method = "FC" if args.Marker_Method == None else args.Marker_Method
        with Spinner():
            MarginalRegression(Count_Data,DNAm_cleaned,filePath=f"{OutputFolder}/CpGs_Marker_{Marker_Method}.txt", by = Marker_Method)
        Methyl_marker_File = pd.read_table(f"{OutputFolder}/CpGs_Marker_{Marker_Method}.txt", index_col=0)
        Methyl_marker = getTopK_commonfeatures(DNAm_Marker_num, Methyl_marker_File, by_pvalue=Marker_Method=="P")
    else:
        Methyl_marker_File = pd.read_table(args.DNAm_Marker, index_col=0)
        if Methyl_marker_File.shape[1] == 0:
            print("DNAm Marker list was input")
            Methyl_marker = Methyl_marker_File.index.tolist()
            if len(Methyl_marker) > DNAm_Marker_num:
                Methyl_marker = Methyl_marker[:DNAm_Marker_num]
        else:
            print("DNAm Marker association matrix was input")
            if args.Marker_Method==None:
                raise ValueError("Please specify Marker_Method")
            Methyl_marker = getTopK_commonfeatures(DNAm_Marker_num, Methyl_marker_File, by_pvalue=args.Marker_Method=="P")
    CommonGene = GEP_Data.index.intersection(GEP_marker)
    CommonCpG = DNAm_Data.index.intersection(Methyl_marker)
    generate_custom_message("Markers selection finished")
    #===============================================================================
    # Model training
    if args.device == "detect":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    aug = 0 if args.Data_augmentation =="No" else args.Data_augmentation
    # define training data
    Training_Data = Train_Tri_Data(GEP_Data, DNAm_Data, Count_Data,
                                   GEP_transform=args.RNA_transform, Methyl_transform=args.DNAm_transform,
                                   by_sample=not args.transform_by_feature, Augmentation=aug, 
                                   scaleCT=args.scale_cellcounts,
                                   GEP_marker=CommonGene, Methyl_marker=CommonCpG, device=device)

    # specify model structure
    with open(args.Model,"r") as json_file:
        structure = json.load(json_file)
    structure["GEP_extract"] = [len(CommonGene)]+structure["GEP_extract"]
    structure["Methyl_extract"] = [len(CommonCpG)]+structure["Methyl_extract"]
    torch.manual_seed(args.seed)
    generate_custom_message("Start training")
    print("Train model on", device, "...\n")
    Model = MOFUN_CCC(structure, loss=args.Loss,
                      activation=args.Activation, dropout1st=args.Dropout).to(device)
    # start training
    Model_Stats = Model.runExp(Training_Data, lr=args.Learning_rate, 
                               batch_size=round(Training_Data.GEP.shape[0]/args.Batch_num), 
                               epochs=args.Epochs, optimizer="Adam")
    Model_save(Model,Training_Data,folderPath=OutputFolder)
    pickle_save(Model_Stats, f"{OutputFolder}/Model_Train_Stats.pkl")

    generate_custom_message("Model Training finished")
    return None

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    print(Logo)
    print("Author: Molin Yue, Email: moy6@pitt.edu, Last updated: 2023-09-25")
    main(args)
    # args = argparse.Namespace(Count = "/ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/CTcount.txt",
    #                           RNA="/ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/rawTPM.txt",
    #                           DNAm="/ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/rawBeta.txt",
    #                           GEP_Marker = "./test/Gene_Marker_FC.txt",
    #                           DNAm_Marker = "./test/CpGs_Marker_FC.txt",
    #                           Marker_Method = "FC",
    #                           Output="./test")
# Test Run
# python Main_train.py --Count /ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/CTcount.txt --RNA /ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/rawTPM.txt --DNAm /ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/rawBeta.txt --Output ./test
# python Main_train.py --Count /ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/CTcount.txt --RNA /ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/rawTPM.txt --DNAm /ix1/wchen/Molin/Research/Paper_Prep/Data/EVAPR_Blood/rawBeta.txt --Output ./test --GEP_Marker ./test/Gene_Marker_FC.txt --DNAm_Marker ./test/CpGs_Marker_FC.txt --Marker_Method FC
