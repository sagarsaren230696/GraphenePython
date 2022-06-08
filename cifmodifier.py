from lib2to3.pytree import convert
from msilib.schema import CreateFolder
from os import sep
from typing import Iterable, List
from venv import create
import numpy as np
import pandas as pd

def readFile(fileName) -> Iterable[str]:
    with open(fileName) as f:
        inputContents = f.readlines()
    return inputContents

def findLastIndex(data,substr:str) -> int:
    dataRev = data[::-1]
    for count,line in enumerate(dataRev):
        if substr in line:
            return len(data) - count
    return -1

def getColumns(data) -> Iterable[str]:
    startIdx = findLastIndex(data,"loop_")
    endIdx = findLastIndex(data,"_atom")
    columnList = data[startIdx:endIdx]
    newColList = [colData.replace('\n','') for colData in columnList]
    return newColList

def unitCellDimension(data) -> Iterable:
    x,y,z = [float(i.split()[1]) for i in data[6:9]]
    return [x,y,z]

def createDf(data) -> pd.DataFrame:
    headers = getColumns(data)
    endIdx = findLastIndex(data,"_atom")
    finalData = data[endIdx:]
    finalData = [dat for dat in finalData if dat != '\n']
    finalData = [colData.replace('\n','') for colData in finalData]
    dataDict = {}
    for count,head in enumerate(headers):
        dataDict[head] = []
        for line in finalData:
            lineDat = line.split()
            dataDict[head].append(lineDat[count])
    df = pd.DataFrame(dataDict,columns=headers)
    return df

def removeLayer(df:pd.DataFrame,columnName:str,condVal:float):
    newDf = df[df[columnName] != str(condVal)]
    return newDf

def changeLayerPos(df:pd.DataFrame,columnName:str,changeVal:List[float]):
    oldVal, newVal = changeVal
    df[columnName] = df[columnName].replace(oldVal,newVal)
    return df

def createNewData(df,origData:List[str]):
    data = df.values

    data = [list(dat)+['\n'] for dat in data]
    data = ['        '.join(dat) for dat in data]
    endIdx = findLastIndex(origData,"_atom")
    firstPart = origData[0:endIdx]
    newData = firstPart+data
    return newData

def changeUnitCellParams(data:Iterable[str],newParams:List[float]):
    firstPart = data[:6]
    secondPart = data[6:13]
    thirdPart = data[13:]
    x,y,z = newParams
    for i,param in enumerate(newParams):
        print(secondPart[i].split())
        secondPart[i] = "    ".join([secondPart[i].split()[0],str(param),'\n'])
    secondPart[-1] = "      ".join([secondPart[-1].split()[0],"{:.2f}".format(x*y*z),'\n'])
    finalData = firstPart+secondPart+thirdPart
    return finalData

def writeFile(fileName,newData:Iterable[str]):
    with open(fileName,"w") as f:
        f.writelines(newData)

def fractToCart(df:pd.DataFrame,dimList:List[float]):
    for i,dim in enumerate(dimList):
        df.iloc[:,i+2] = pd.to_numeric(df.iloc[:,i+2])*np.float64(dim)
    return df

def cartToFract(df:pd.DataFrame,dimList:List[float]):
    for i,dim in enumerate(dimList):
        df.iloc[:,i+2] = df.iloc[:,i+2]/dim
    return df

def modifyLength(df:pd.DataFrame,currentDim:List[float],newDim:List[float]):
    dfInCart = fractToCart(df,currentDim)
    df_C1 = dfInCart[(dfInCart["_atom_site_label"]=="C1")]
    x_C1 = df_C1["_atom_site_fract_x"].unique()
    d_C1_x = np.diff(x_C1[:2])[0]
    y_C1 = df_C1["_atom_site_fract_y"].unique()
    d_C1_y = np.diff(y_C1[:2])[0]

    df_C2 = dfInCart[(dfInCart["_atom_site_label"]=="C2")]
    x_C2 = df_C2["_atom_site_fract_x"].unique()
    maxDf_C1C3_x = pd.DataFrame(dfInCart[dfInCart["_atom_site_fract_x"]==max(x_C1)])
    maxDf_C2C4_x = pd.DataFrame(dfInCart[dfInCart["_atom_site_fract_x"]==max(x_C2)])
    maxDf_C1C3_x["_atom_site_fract_x"] = maxDf_C1C3_x["_atom_site_fract_x"] + d_C1_x
    maxDf_C2C4_x["_atom_site_fract_x"] = maxDf_C2C4_x["_atom_site_fract_x"] + d_C1_x

    dfInCartNew = pd.concat([dfInCart,maxDf_C1C3_x,maxDf_C2C4_x])
    df_new = cartToFract(dfInCartNew,newDim)

    df_new = df_new.round({"_atom_site_fract_x":6,"_atom_site_fract_y":6,"_atom_site_fract_z":6})
    convert_dict = {"_atom_site_fract_x":str,"_atom_site_fract_y":str,"_atom_site_fract_z":str}
    df_new.iloc[:,2:5]=df_new.iloc[:,2:5].astype(convert_dict)

    # print(df_new.dtypes)
    return df_new

    

# cifData = readFile("graphite-sheet-8.52A.cif")
# df = createDf(cifData)
# ## For double layer
# newDf = removeLayer(df,"_atom_site_fract_z",0.666667)
# newCifData = createNewData(newDf,cifData)
# writeFile("graphite-sheet-double_layer.cif",newCifData)
# ## For single layer
# newDf = removeLayer(newDf,"_atom_site_fract_z",0.333333)
# newCifData = createNewData(newDf,cifData)
# writeFile("graphite-sheet-single_layer.cif",newCifData)

# ## Keep the layer in middle
# newDf = changeLayerPos(newDf,"_atom_site_fract_z",["0","0.5"])
# newCifData = createNewData(newDf,cifData)
# writeFile("graphite-sheet-single_layer-0.5.cif",newCifData)

### Increasing unit cell length along x axis
cifData = readFile("graphite-sheet-single_layer.cif")
currentDim = unitCellDimension(cifData)
df = createDf(cifData)
newDf = modifyLength(df,currentDim,[41.82,25.56,25.56])
newCifData = createNewData(newDf,cifData)
finalCifData = changeUnitCellParams(newCifData,[41.82,25.56,25.56])
writeFile("graphite-sheet-single_layer_increased_x.cif",finalCifData)