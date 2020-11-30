import numpy as np
import pandas as pd
import os


###################################
# 01.get all data file full path(files<type:list>)
### - get data size
###################################
rootDir = 'D:/DeepLearning/GAR/1202/AFAD-Full'
files = []
for (dirpath, dirnames, filenames) in os.walk(rootDir):
    #.\AFAD-Full
    #.\AFAD-Full\18 
    #.\AFAD-Full\18\111
    for file in filenames:
        if file.endswith('.jpg'):
            #.\AFAD-Lite\18\111\100062-0.jpg
            #18\111\100062-0.jpg
            files.append(os.path.relpath(os.path.join(dirpath,file),rootDir))
print("Data size:",len(files))
print("Data form:",files[0])


###################################
# 02.trun to dataframe(dataFrame<type:pd>)
### - get data range
### - process gen data
### - get age/gen datasize(dataInfo<type:pd>)
###################################
attri = {}
attri['age'] = []
attri['gender'] = []
attri['genID']=[]
attri['file'] = []
attri['path'] = []

for f in files:
    age, gender, fname = f.split('\\')
    if gender == '111':
        gender = 'male'
        genID = 0
    else:
        gender = 'female'
        genID = 1
        
    attri['age'].append(age)
    attri['gender'].append(gender)
    attri['genID'].append(genID)
    attri['file'].append(fname)
    attri['path'].append(f)

dataFrame = pd.DataFrame.from_dict(attri)
#print(dataFrame.head())
#print("Data range: min=",dataFrame['age'].min(),",max=",dataFrame['age'].max())

dataInfo = dataFrame.groupby(['age','genID']).count()
dataInfo = dataInfo.drop(['gender','path'],axis=1).rename(columns={'file':'count'})
#print(dataInfo.head())
#show
plot = dataInfo.plot.bar()

#######################################
#03.process age data
#######################################
#print(type(dataFrame['age'].min().astype(int)))
dataFrame = dataFrame.assign(ageID=dataFrame['age'].values.astype(int) - int(dataFrame['age'].min()))
ageNum = np.unique(dataFrame['ageID'].values).shape[0]
#print("Age Num:",ageNum,dataFrame['ageID'].max)
#print(dataFrame.head())
#print(dataFrame.dtypes)


#######################################
#04.sample train/test
### - random
#######################################
np.random.seed(123)
index = np.random.rand(len(dataFrame)) < 0.8
dfTrain = dataFrame[index]
dfTest = dataFrame[~index]
#print("Test dataSize:",len(dfTest),"/Train dataSize:",len(dfTrain))

########################################
#05.save to csv file
########################################
#dfTrain.set_index('file', inplace=True)
#dfTrain.to_csv('training_set.csv')
#dfTest.set_index('file', inplace=True)
#dfTest.to_csv('testing_set.csv')
#dataInfo.to_csv('dataInfo.csv')


