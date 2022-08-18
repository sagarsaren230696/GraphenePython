from lib2to3.pytree import convert
from msilib.schema import CreateFolder
from os import sep
from typing import Iterable, List
from venv import create
import numpy as np
import pandas as pd
import math

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
    """For single layer"""
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
    """Returns df in float"""
    dfNew = df.copy()
    for i,dim in enumerate(dimList):
        dfNew.iloc[:,i+2] = pd.to_numeric(dfNew.iloc[:,i+2])*np.float64(dim)
    return dfNew

def cartToFract(df:pd.DataFrame,dimList:List[float]):
    for i,dim in enumerate(dimList):
        df.iloc[:,i+2] = df.iloc[:,i+2]/dim
    return df

def increaseXLen(dfInCart:pd.DataFrame,d_C1):
    colHeader = "_atom_site_fract_x"
    df_C1 = dfInCart[(dfInCart["_atom_site_label"]=="C1")]
    pos_C1 = df_C1[colHeader].unique()
    df_C2 = dfInCart[(dfInCart["_atom_site_label"]=="C2")]
    pos_C2 = df_C2[colHeader].unique()
    maxDf_C1C3 = pd.DataFrame(dfInCart[dfInCart[colHeader]==max(pos_C1)])
    maxDf_C2C4 = pd.DataFrame(dfInCart[dfInCart[colHeader]==max(pos_C2)])
    maxDf_C1C3[colHeader] = maxDf_C1C3[colHeader] + d_C1
    maxDf_C2C4[colHeader] = maxDf_C2C4[colHeader] + d_C1
    dfInCartNew = pd.concat([dfInCart,maxDf_C1C3,maxDf_C2C4])
    return dfInCartNew

def increaseYLen(dfInCart:pd.DataFrame,d_C1):
    colHeader = "_atom_site_fract_y"
    Carr = [f"C{i}" for i in [1,2,3,4]]
    dfInCartNew = dfInCart.copy()
    for Cval in Carr:
        df_C = dfInCart[(dfInCart["_atom_site_label"]==Cval)]
        maxPos = max(df_C[colHeader].unique())
        maxDf = pd.DataFrame(dfInCart[dfInCart[colHeader]==maxPos])
        maxDf[colHeader] = maxDf[colHeader] + d_C1
        dfInCartNew = pd.concat([dfInCartNew,maxDf])
    return dfInCartNew

def decreaseLen(dfInCart:pd.DataFrame,direction='x'):
    dfInCartNew = dfInCart.copy()
    colHeader = "_atom_site_fract_{}".format(direction)
    df_C1 = dfInCartNew[(dfInCartNew["_atom_site_label"]=="C1")]
    pos_C1 = df_C1[colHeader].unique()
    df_C2 = dfInCartNew[(dfInCartNew["_atom_site_label"]=="C2")]
    pos_C2 = df_C2[colHeader].unique()
    dfInCartNew = dfInCartNew.drop(dfInCartNew[dfInCartNew[colHeader]==max(pos_C1)].index)
    dfInCartNew = dfInCartNew.drop(dfInCartNew[dfInCartNew[colHeader]==max(pos_C2)].index)
    return dfInCartNew 

def modifyLength(df:pd.DataFrame,currentDim:List[float],newDim:List[float]):
    dfInCart = fractToCart(df,currentDim)
    df_C1 = dfInCart[(dfInCart["_atom_site_label"]=="C1")]
    x_C1 = df_C1["_atom_site_fract_x"].unique()
    d_C1_x = np.diff(x_C1[:2])[0]
    y_C1 = df_C1["_atom_site_fract_y"].unique()
    d_C1_y = np.diff(y_C1[:2])[0]

    # print(d_C1_y)
    dfInCartNew = dfInCart
    nx = int((newDim[0]-currentDim[0])/d_C1_x)
    if nx > 1:
        for n in range(nx):
            dfInCartNew = increaseXLen(dfInCartNew,d_C1_x)
    elif nx < 0:
        for n in range(abs(nx)):
            dfInCartNew = decreaseLen(dfInCartNew)

    ny = int((newDim[1]-currentDim[1])/round(d_C1_y))
    if ny > 1:
        for n in range(ny):
            dfInCartNew = increaseYLen(dfInCartNew,d_C1_y)
    elif ny < 0:
        for n in range(abs(ny)):
            dfInCartNew = decreaseLen(dfInCartNew,direction='y')
    
    df_new = cartToFract(dfInCartNew,newDim)
    df_new = df_new.round({"_atom_site_fract_x":6,"_atom_site_fract_y":6,"_atom_site_fract_z":6}) # round all elements column-wise
    convert_dict = {"_atom_site_fract_x":str,"_atom_site_fract_y":str,"_atom_site_fract_z":str}
    df_new.iloc[:,2:5]=df_new.iloc[:,2:5].astype(convert_dict)
    return df_new

def addLayers(df:pd.DataFrame,numOfLayers:int,spacing:float,zLen:float)->pd.DataFrame:
    """Apply to single layered graphene"""
    zFractPos = [(i+1)*spacing/zLen for i in range(numOfLayers)]
    dfNewList = [df]
    for count, zPos in enumerate(zFractPos):
        dfTemp = df.copy()
        dfTemp["_atom_site_fract_z"] = str(round(zPos,6))
        dfNewList.append(dfTemp)
    df_new = pd.concat(dfNewList,axis=0)
    return df_new

def createMiddlePore(df:pd.DataFrame,zLen,cutOff:float,poreSize:float) -> pd.DataFrame:
    bottomDf = df.copy()
    topDf = df.copy()
    bottomDf["_atom_site_fract_z"] = bottomDf["_atom_site_fract_z"].apply(lambda val: str(round(float(val)+ cutOff/zLen,6))) 
    maxZbottom = max([float(i) for i in bottomDf["_atom_site_fract_z"].unique()])
    print(maxZbottom)
    topDf["_atom_site_fract_z"] = topDf["_atom_site_fract_z"].apply(lambda val: str(round(float(val)+ maxZbottom +poreSize/zLen,6))) 
    dfNewList = [bottomDf,topDf]
    dfNew = pd.concat(dfNewList,axis=0)
    return dfNew

def addFunctionalGroups(df:pd.DataFrame,dims:List[float],fg:List[str],fgDims:List[float],fgCharge:List[float])->pd.DataFrame:
    df_cart=fractToCart(df,dims) 
    layers = df_cart["_atom_site_fract_z"].apply(lambda val: round(float(val),1)).unique()[2:4]
    fg_layers = [layers[0]+fgDims[0],layers[1]-fgDims[0]] # For Carbonyl
    def find_nearest(array, value):
        # array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return [idx,array[idx]]
    ## At fixed y
    yLoc = find_nearest(df['_atom_site_fract_y'].apply(lambda val: float(val)).values,0.5)[1]
    fgBase = df_cart.loc[np.isclose(df['_atom_site_fract_y'].apply(lambda val: float(val)),yLoc)]
    print(fgBase['_atom_site_fract_z'].unique())
    fgBase1 = fgBase.loc[np.isclose(fgBase['_atom_site_fract_z'].apply(lambda val: round(val,1)),layers[0])]
    fgBase2 = fgBase.loc[np.isclose(fgBase['_atom_site_fract_z'].apply(lambda val: round(val,1)),layers[1])]
    fgBase1.loc[:,"_atom_site_fract_z"] = round(fg_layers[0],6)
    fgBase2.loc[:,"_atom_site_fract_z"] = round(fg_layers[1],6)
    fgBase = [fgBase1,fgBase2]
    fgBase = pd.concat(fgBase,axis=0)
    fgBase.iloc[:,0] = fg[0]
    fgBase.iloc[:,1] = fg[1]
    
    # delete alternate rows 
    fgBase = fgBase.iloc[::2]
    print(fgBase['_atom_site_fract_z'].shape)

    # change charge
    fgBase.iloc[:,-1] = str(round(fgCharge[1],4))
    for index, row in fgBase.iterrows():
        print(index)
        df_cart.iloc[index,-1] = str(round(fgCharge[0],4))
        df_cart.iloc[index,0] = "C_FG"

    df_cart = [df_cart,fgBase]
    df_cart = pd.concat(df_cart,axis=0)
    df_new = cartToFract(df_cart,dims)
    print(df_new["_atom_site_fract_z"].unique())
    convert_dict = {"_atom_site_fract_x":str,"_atom_site_fract_y":str,"_atom_site_fract_z":str}
    df_new.iloc[:,2:5]=df_new.iloc[:,2:5].astype(convert_dict)

    return df_new


def poreBlockGenerator(dims:List[float],nLayers:int,spacing:float,poreSize:float=7):
    blockSphereRadius = spacing/2
    xSpheres = np.asarray([i for i in np.arange(blockSphereRadius,dims[0],blockSphereRadius)])
    ySpheres = np.asarray([i for i in np.arange(blockSphereRadius,dims[1],blockSphereRadius)])
    
    xSpheres,ySpheres = np.meshgrid(xSpheres,ySpheres)
    with open(f"graphite-sheet_{nLayers}-layers_{poreSize}A.block","w") as f:
        f.write(f"{xSpheres.shape[0]*xSpheres.shape[1]*(nLayers-1)}\n")
        for n in range(nLayers-1):
            for xVal,yVal in zip(xSpheres,ySpheres):
                lines = [f"{x/dims[0]:.3f}\t{y/dims[1]:.3f}\t{(spacing/2+n*spacing)/dims[2]:.3f}\t{spacing/2:.3f}\n" for x,y in zip(xVal,yVal)]
                f.writelines(lines)


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
# cifData = readFile("graphite-sheet-single_layer.cif")
# currentDim = unitCellDimension(cifData)
# df = createDf(cifData)
# poreSizes = [8.9,18.5,27.9]
# # dims = [int(60/2.46)*2.46,int(60/4.26)*4.26,round(27.9+3.35*2,2)]
# for poreSize in poreSizes:
#     dims = [round(int(40/2.46)*2.46,2),round(int(40/4.26)*4.26,2),round(poreSize,2)] #+3.35*2
#     newDf = modifyLength(df,currentDim,dims) ## default is 16*2.46 and 6*4.26
#     # newestDf = addLayers(newDf,2,3.35,dims[2])
#     newCifData = createNewData(newDf,cifData)
#     finalCifData = changeUnitCellParams(newCifData,dims)
#     writeFile(f"graphite-sheet_single-layers_{poreSize}A.cif",finalCifData)

# poreBlockGenerator([round(int(40/2.46)*2.46,2),round(int(40/4.26)*4.26,2),round(7+3.35*2,2)],3,3.35)

# cutOff,spacing,numOfLayers,poreSize = [9,3.35,3,7]
# cifData = readFile("graphite-sheet_3-layers_7A.cif")
# currentDim = unitCellDimension(cifData)
# newZ = cutOff*2+spacing*(numOfLayers-1)*2+poreSize
# newDim = currentDim[:2] + [newZ]
# df = createDf(cifData)
# df["_atom_site_fract_z"] = df["_atom_site_fract_z"].apply(lambda z:str(float(z)*currentDim[2]/newDim[2]))
# newMiddlePoreDf = createMiddlePore(df,newZ,cutOff,poreSize)
# newCifData = createNewData(newMiddlePoreDf,cifData)
# finalCifData = changeUnitCellParams(newCifData,newDim)
# writeFile("graphite-sheet_3-layers_7A_middlePore.cif",finalCifData)

### Add functional groups
cifData = readFile("graphite-sheet_3-layers_7A_middlePore.cif")
currentDim = unitCellDimension(cifData)
df = createDf(cifData)
newDf = addFunctionalGroups(df,currentDim,["O_CO","O"],[1.433],[0.5,-0.5])
newCifData = createNewData(newDf,cifData)
writeFile("graphite-sheet_3-layers_7A_middlePore_FG-CO.cif",newCifData)



### Creating multilayer graphite
# cifData = readFile("graphite-sheet-single_layer.cif")
# currentDim = unitCellDimension(cifData)
# df = createDf(cifData)
# dims = [39.36,38.34,3.35*2+40]
# newDf = modifyLength(df,currentDim,dims) ## default is 16*2.46 and 6*4.26
# newestDf = addLayers(newDf,2,3.35,dims[2])
# newCifData = createNewData(newestDf,cifData)
# finalCifData = changeUnitCellParams(newCifData,dims)
# writeFile(f"graphite-sheet_3-layers_40A.cif",finalCifData)