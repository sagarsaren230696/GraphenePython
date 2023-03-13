import sys
from typing import Iterable, List
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def readFile(fileName) -> Iterable[str]:
    with open(fileName) as f:
        inputContents = f.readlines()
    return inputContents

def writeFile(fileName,newData:Iterable[str]):
    with open(fileName,"w") as f:
        f.writelines(newData)

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
    newColList = [colData.replace('\n','').split()[0] for colData in columnList] # split()[0] added to remove whitespace from the str
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

def createNewData(df,origData:List[str]):
    data = df.values

    data = [list(dat)+['\n'] for dat in data]
    data = ['        '.join(dat) for dat in data]
    endIdx = findLastIndex(origData,"_atom")
    firstPart = origData[0:endIdx]
    newData = firstPart+data
    return newData

def chargeModify(dataDf:pd.DataFrame,typeOfChange:str="modify",modifier:float=1.0):
    print(dataDf.columns)
    chargeList = list(pd.to_numeric(dataDf["_atom_site_charge"]))
    print(type(modifier))
    if typeOfChange == "modify":
        newChargeList = [charge*modifier for charge in chargeList]
    newDf = dataDf.copy()
    newDf["_atom_site_charge"] = [str(newCharge) for newCharge in newChargeList]
    return newDf 

def labelModify(dataDf:pd.DataFrame,appendNums:bool=True):
    if appendNums:
        uniqueLabels = dataDf["_atom_site_label"].unique()
        newDf = dataDf.sort_values(by='_atom_site_label',ignore_index=True)
        labelCounts = newDf["_atom_site_label"].value_counts().sort_index().to_dict()
        # print(newDf["_atom_site_label"].value_counts().sort_index())
        oldPos = 0
        for key,value in labelCounts.items():
            newStr= [key+str(i+1) for i in range(value)]
            print(key,oldPos,oldPos+value)
            # dfIdx = [i for i in range(oldPos,oldPos+value)]
            for idx in range(oldPos,oldPos+value):
                newDf.iloc[idx,0] = newStr[idx-oldPos]
            # newDf = newDf.replace(dfIdx,newStr)
            oldPos = oldPos+value
    # print(labelCounts.to_dict())
    return newDf

### Increasing unit cell length along x axis
if __name__=="__main__":
    """Generate pore with a single wall containing multiple layers of graphene sheets in a unit cell"""
    if len(sys.argv) != 3:
        sys.exit("Please provide fileName and Modifier")
    fileName = sys.argv[1]
    modifier = float(sys.argv[2])
    print(fileName)
    cifData = readFile(fileName)
    df = createDf(cifData)
    # print(df.head())
    newDf_labelModified = labelModify(df)
    newDf_chargeModified = chargeModify(newDf_labelModified,modifier=modifier)
    newCifData = createNewData(newDf_chargeModified,cifData)
    writeFile(fileName.split(".cif")[0]+"x"+str(modifier)+".cif",newCifData)
