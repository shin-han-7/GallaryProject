import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module

import numpy as np
import pandas as pd
from PIL import Image
import time

#####################
# setting
#####################
TRAIN_CSV_PATH = 'training_set.csv'
TEST_CSV_PATH = 'testing_set.csv'
VALID_CSV_PATH = 'validing_set.csv'
ROOT_DIR = 'D:/DeepLearning/GAR/1202/AFAD-Full'
TRAIN_PATH = 'train'
TEST_PATH = 'test'
VALID_PATH = 'valid'
BATCH_SIZE = 80
NUM_WORKERS = 0

cuda = 0
if cuda >= 0:
    DEVICE = torch.device("cuda:%d" % cuda)
else:
    DEVICE = torch.device("cpu")

train_transform = transforms.Compose([
	transforms.Scale(256),
	transforms.RandomCrop(227),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()
	])

test_transform = transforms.Compose([
	transforms.Scale(256),
	transforms.CenterCrop(227),
	transforms.ToTensor()
	])
train_data = dsets.ImageFolder(root='train',transform=train_transform)
test_data = dsets.ImageFolder(root='test',transform=test_transform)
print(type(train_data),train_data.class_to_idx)

##########################
# DataLoad
##########################
'''files = []
for (dirpath, dirnames, filenames) in os.walk(ROOT_DIR):
    #.\AFAD-Full
    #.\AFAD-Full\18 
    #.\AFAD-Full\18\111
    for label in dirnames:
        files.append()


class AFADDatasetGen(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.gen = df['genID'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, self.img_paths[index]))
        if self.transform is not None:
            img = self.transform(img)
            
        target = self.gen[index]
        #target = torch.tensor(target, dtype=torch.float32)
        return img, target

    def __len__(self):
        return self.gen.shape[0]
 


train_transform = transforms.Compose([transforms.Scale(256),
                                      transforms.RandomCrop(227),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                      ])

train_dataset = AFADDatasetGen(csv_path=TRAIN_CSV_PATH, 
                               root_dir=ROOT_DIR,
                               transform= train_transform) 

train_loader = DataLoader(train_dataset,
                          batch_size= BATCH_SIZE,
                          shuffle=True,
                          num_workers= NUM_WORKERS)

test_transform = transforms.Compose([transforms.Scale(256),
                                     transforms.CenterCrop(227),
                                     transforms.ToTensor()
                                     ])

test_dataset = AFADDatasetGen(csv_path=TRAIN_CSV_PATH, 
                              root_dir=ROOT_DIR,
                              transform= test_transform) 

test_loader = DataLoader(test_dataset,
                         batch_size= BATCH_SIZE,
                         shuffle=False,
                         num_workers= NUM_WORKERS)

#####################
# Model
#####################
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(3, 16, kernel_size=5, stride=1) 
        self.relu1 = nn.ReLU(inplace=True) 
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(16,8, kernel_size=11, stride=1) 
        self.relu2 = nn.ReLU(inplace=True) 
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 ,#input_shape=(8*50*50)
        self.fc = nn.Linear(8 * 50 * 50, 2)     
    
    def forward(self, x):
        out = self.cnn1(x) 
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1) 
        out = self.fc(out) 
        return out

######################
# Init train info
######################
num_epochs = 1#100
epoch_history = []
loss_history = []
train_acc_history = []
valid_acc_history = []
best_valid_acc = 0.0
learning_rate = 0.001

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = CNN()
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)

'''
########
# training
#########
