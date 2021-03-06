import numpy as np
import pandas as pd
import os
import  argparse

parser = argparse.ArgumentParser()
#local
#parser.add_argument('--rootDir', type=str, help='get dataset dirRoot(path)', default='./AFAD-Full')
#parser.add_argument('--maleFileName', type=str, help='age/gen dataset gen file name', default='111')
parser.add_argument('--rootDir', type=str, help='get dataset dirRoot(path)', required = True)
parser.add_argument('--maleFileName', type=str, help='age/gen dataset gen file name', required = True)
parser.add_argument('--traincsv', type=str, help='train csv file rename', default='training_set_01.csv')
parser.add_argument('--testcsv', type=str, help='test csv file rename', default='testing_set_01.csv')
parser.add_argument('--validcsv', type=str, help='valid csv file rename', default='validing_set_01.csv')
parser.add_argument('--info', type=str, help='dataset info csv file rename', default='info_01.csv')
args = parser.parse_args()
rootDir = args.rootDir
maleFileName = args.maleFileName

###################################
# 01.get all data file full path(files<type:list>)
### - get data size
###################################
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
    if gender == maleFileName:
        gender = 'male'
        genID = 0
    else:
        gender = 'female'
        genID = 1
        
    attri['age'].append(age)
    attri['gender'].append(gender)
    attri['genID'].append(genID)
    attri['file'].append(fname)
    attri['path'].append(age+'/'+gen+'/'+fname)

dataFrame = pd.DataFrame.from_dict(attri)
#print(dataFrame.head())
print("Data range: min=",dataFrame['age'].min(),",max=",dataFrame['age'].max())

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
#print("Age Num:",ageNum,",ageID max:",dataFrame['ageID'].max())
#print(dataFrame.head())
print(dataFrame.dtypes)


#######################################
#04.sample train/test
### - random
#######################################
np.random.seed(123)
probs = np.random.rand(len(dataFrame))
train_msk = probs < 0.8
test_msk = (probs>=0.8) & (probs < 0.9)
valid_msk = probs >= 0.9
dfTrain = dataFrame[train_msk]
dfTest = dataFrame[test_msk]
dfValid = dataFrame[valid_msk]
print("len train,vaild,test = ",len(dfTrain),len(dfValid),len(dfTest))

########################################
#05.save to csv file
########################################
train_csv = args.traincsv
dfTrain.set_index('file', inplace=True)
dfTrain.to_csv(train_csv)

test_csv = args.testcsv
dfTest.set_index('file', inplace=True)
dfTest.to_csv(test_csv)

valid_cvs = args.validcsv
dfValid.set_index('file', inplace=True)
dfValid.to_csv(valid_csv)

info_csv = args.info
dataInfo.to_csv(info_csv)


