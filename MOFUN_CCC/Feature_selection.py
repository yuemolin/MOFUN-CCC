import statsmodels.api as sm
import pandas as pd
from Data_prep import getDfRange
import warnings

def MarginalRegression(CT,Omics,filePath,by = "FC"):
    # CT: pd.DataFrame, index = samples, columns = cells
    # Omics: pd.DataFrame, index = genes, columns = cells
    # filePath: str, path to save the output
    # by: str, "pvalue" or "FC"
    assert by in ["pvalue","FC"], "Choose either pvalue or FC"
    assert (CT.index==Omics.columns).all, "CT and Omics have different samples"
    if max([abs(x) for x in getDfRange(Omics)]) >50:
        warnings.warn("Input data have large (|x|>50) value, please check")
    assert all(Omics.var(axis=1) != 0) , "Input data have zero variance, please check"
    OutMat = pd.DataFrame(index=Omics.index,columns=CT.columns)
    for cells in CT.columns:
        y = CT.loc[:,cells]
        for genes in Omics.index:
            # print(cells,",",genes)
            x = Omics.loc[genes,:]
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            if by == "pvalue":
                OutMat.at[genes, cells] =  model.pvalues[1]
            else:
                OutMat.at[genes, cells] =  model.params[1]
    OutMat.to_csv(filePath,sep="\t")

# def getRegression_res(CT,Omics, filePath,pvalue = True,verbose=False):
#     if os.path.exists(filePath):
#         EVAPR_GEP_p = pd.read_table(filePath, index_col=0)
#     else:
#         EVAPR_GEP_p = MarginalRegression(CT, Omics, filePath=filePath,pvalue=pvalue)
#     return EVAPR_GEP_p

def getTopK_commonfeatures(K,*Res,by_pvalue,verbose=False):
    # K: int, number of features to select
    # Res: MarginalRegression output, marker matrix by pvalue or FC, 
    #      can input multiple for taking interactions
    # by_pvalue: bool, True: select marker by pvalue, False: select marker by FC
    # verbose: bool, 
    # return: list, common features
    studies_list = []
    for study_i in Res:
        v_min,v_max  = getDfRange(study_i)
        # by_pvalue = False
        if v_max < 1 and v_min >0 :
            by_pvalue = True
        print("select marker by pvalue") if by_pvalue else print("select marker by FC")
        target = set()
        study_i = study_i.abs()
        for names in study_i.columns:
            if by_pvalue:
                target |= set(study_i.nsmallest(K,names,keep = 'first').index)
            else:
                target |= set(study_i.nlargest(K,names,keep = 'first').index)
        studies_list.append(target)
    common = set.intersection(*map(set,studies_list))
    if verbose:
        print("#common features: ",len(common))
    return sorted(list(common))
