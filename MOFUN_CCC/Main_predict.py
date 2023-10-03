import argparse
import pandas as pd
from utils import Logo
from DLModel import Model_load
from Data_prep import Test_Tri_Data

def get_args_parser():
    parser = argparse.ArgumentParser('MOFUN-CCC Cell Count predicttion ', add_help=False)
    # Input/Output parameters
    parser.add_argument('--RNA', type=str,
                        help='tab seperated txt gene TPM file (rows: Genes, columns: Samples)')
    parser.add_argument('--DNAm', type=str,
                        help='tab seperated txt CpG Beta file (rows: CpGs, columns: Samples)')
    parser.add_argument('--Output', type=str,required=True,
                        help='Output filename')
    #===============================================================================
    # Model parameters
    parser.add_argument('--Model_Path', type=str,required=True,
                        help='Saved Model path, at least contains Model_str.pkl, Model_dict.pt, and TrainShell.pkl')
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Show this help message and exit')

    #===============================================================================
    return parser

def main(args):
    device = "cpu"
    TrainShell,Model = Model_load(folderPath=args.Model_Path, device="cpu")
    GEP_Data = DNAm_Data = None
    if args.RNA == None and args.DNAm == None:
        raise ValueError("Input at least one modality of data")
    if args.RNA != None:
        GEP_Data = pd.read_table(args.RNA, index_col=0)
    if args.DNAm != None:
        DNAm_Data = pd.read_table(args.DNAm, index_col=0)
    if args.RNA != None and args.DNAm != None:
        commonID = GEP_Data.columns.intersection(DNAm_Data.columns)
        GEP_Data = GEP_Data[commonID]
        DNAm_Data = DNAm_Data[commonID]
        print(len(commonID)," Common samples were predicted" )
    Test_Data = Test_Tri_Data(Train_Tri_Data=TrainShell, GEP=GEP_Data,Methyl=DNAm_Data, device=device)
    Pred_Count = Test_Data.predict_CT(model=Model).cpu()
    Test_Data.save_pred_CT(filename=f"{args.Output}")

    return

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # args = argparse.Namespace(RNA="/ix1/wchen/DATA/BulkMultiOmics/Cleaned/13_GEP.txt",
    #                           DNAm="/ix1/wchen/DATA/BulkMultiOmics/Cleaned/13_Methyl.txt",
    #                           Output="/ix1/wchen/Molin/Research/Paper_Prep/PublicData/13_Asthma_PBMC/MOFUN_EVAPR_model_pred.txt",
    #                           Model_Path='/ihome/wchen/moy6/software/MOFUN_CCC/MOFUN_CCC/Models/EVAPR_blood_model')

    print(Logo)
    print("Author: Molin Yue, Email: moy6@pitt.edu, Last updated: 2023-09-25")
    main(args)


# python Main_predict.py --RNA /ix1/wchen/Molin/Research/Paper_Prep/PublicData/01_Covid/01_GEP.txt --DNAm /ix1/wchen/Molin/Research/Paper_Prep/PublicData/01_Covid/01_Methyl.txt --Output ./test/COVID_test.txt --Model_Path ./test/
# python Main_predict.py --RNA /ix1/wchen/Molin/Research/Paper_Prep/PublicData/02_Parkinson/02_GEP.txt --DNAm /ix1/wchen/Molin/Research/Paper_Prep/PublicData/02_Parkinson/02_Methyl.txt --Output ./test/Parkinson_test.txt --Model_Path ./test/