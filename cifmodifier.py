from os import sep
from typing import Iterable, List
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

def modifyDf(df:pd.DataFrame,columnName:str,condVal:float):
    newDf = df[df[columnName] != str(condVal)]
    return newDf

def createNewData(df,origData:List[str]):
    data = df.values

    data = [list(dat)+['\n'] for dat in data]
    data = ['        '.join(dat) for dat in data]
    endIdx = findLastIndex(origData,"_atom")
    firstPart = origData[0:endIdx]
    newData = firstPart+data
    return newData

def writeFile(fileName,newData:Iterable[str]):
    with open(fileName,"w") as f:
        f.writelines(newData)

cifData = readFile("graphite-sheet-8.52A.cif")
df = createDf(cifData)
## For double layer
newDf = modifyDf(df,"_atom_site_fract_z",0.666667)
newCifData = createNewData(newDf,cifData)
writeFile("graphite-sheet-double_layer.cif",newCifData)
## For single layer
newDf = modifyDf(newDf,"_atom_site_fract_z",0.333333)
newCifData = createNewData(newDf,cifData)
writeFile("graphite-sheet-single_layer.cif",newCifData)