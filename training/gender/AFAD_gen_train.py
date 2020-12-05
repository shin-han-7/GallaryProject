import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import numpy as np
import pandas as pd
from PIL import Image
import time
import os
import argparse

#####################
# setting
#####################
TRAIN_CSV_PATH = '../../prepare/training_set.csv'
TEST_CSV_PATH = '../../prepare/testing_set.csv'
VALID_CSV_PATH = '../../prepare/validing_set.csv'
ROOT_DIR = '..\\..\\dataset\\AFAD-Full\\AFAD-Full'

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=int,default=-1)
parser.add_argument('--storepath',type=str,required=True)
parser.add_argument('--epoch',type=int,required=True,help = 'num_epochs')
parser.add_argument('--batch',type=int,required=True , help = 'BATCH_SIZE')
args = parser.parse_args()

LEARN_RATE = 0.001
BATCH_SIZE = args.batch
EPOCH = args.epoch
NUM_WORKERS = 0
STORE_PATH = args.storepath
if not os.path.exists(STORE_PATH):
    os.mkdir(STORE_PATH)
LOGFILE = os.path.join(STORE_PATH, 'training.log')
MODEL_SAVE = os.path.join(STORE_PATH,'model_gen.pt')

cuda = args.cuda
if cuda >= 0:
    DEVICE = torch.device("cuda:%d" % cuda)
else:
    DEVICE = torch.device("cpu")

###################
# logging
###################
header = []
header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Output Path: %s' % STORE_PATH)

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()


##########################
# DataLoad
##########################
class AFAD_Gen_Dataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.gens = df['genID'].values
        #self.gens = df['gender']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, self.img_paths[index]))
        if self.transform is not None:
            img = self.transform(img)
        label = self.gens[index]
        #label = torch.tensor(label, dtype=torch.float32)
        return img, label

    def __len__(self):
        return self.gens.shape[0] #len(self.gens)
 


train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.RandomCrop((227,227)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                      ])

train_dataset = AFAD_Gen_Dataset(csv_path=TRAIN_CSV_PATH, 
                               root_dir=ROOT_DIR,
                               transform= train_transform) 

train_loader = DataLoader(train_dataset,
                          batch_size= BATCH_SIZE,
                          shuffle=True,
                          num_workers= NUM_WORKERS)

test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.CenterCrop((227,227)),
                                     transforms.ToTensor()
                                     ])

test_dataset = AFAD_Gen_Dataset(csv_path=TEST_CSV_PATH, 
                              root_dir=ROOT_DIR,
                              transform= test_transform) 

test_loader = DataLoader(test_dataset,
                         batch_size= BATCH_SIZE,
                         shuffle=False,
                         num_workers= NUM_WORKERS)

valid_dataset = AFAD_Gen_Dataset(csv_path=VALID_CSV_PATH, 
                              root_dir=ROOT_DIR,
                              transform= test_transform) 

valid_loader = DataLoader(valid_dataset,
                         batch_size= BATCH_SIZE,
                         shuffle=False,
                         num_workers= NUM_WORKERS)

#print('[id=10]img arr:',test_dataset[10][0],',genID:',test_dataset[10][1])
#print('type:',type(test_dataset[10][0]),type(test_dataset[10][1]))
#<class 'torch.Tensor'> <class 'numpy.int64'>
#print('len:',len(test_dataset[10][0]))#3
images,labels=next(iter(test_loader))
print(images.shape,labels.shape)


################
# Model
#################
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) #output_shape=(16,220,220) #(224-5+1)/1 #(weigh-kernel+1)/stride 無條件進位
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,110,110) #(220/2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,106,106)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,53,53)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,51,51)
        self.relu3 = nn.ReLU() # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,25,25)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0) #output_shape=(8,23,23)
        self.relu4 = nn.ReLU() # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) #output_shape=(8,11,11)
        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(8 * 11 * 11, 512) 
        self.relu5 = nn.ReLU() # activation
        self.fc2 = nn.Linear(512, 2) 
        self.output = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        out = self.cnn1(x) # Convolution 1
        out = self.relu1(out)
        out = self.maxpool1(out)# Max pool 1
        out = self.cnn2(out) # Convolution 2
        out = self.relu2(out) 
        out = self.maxpool2(out) # Max pool 2
        out = self.cnn3(out) # Convolution 3
        out = self.relu3(out)
        out = self.maxpool3(out) # Max pool 3
        out = self.cnn4(out) # Convolution 4
        out = self.relu4(out)
        out = self.maxpool4(out) # Max pool 4
        out = out.view(out.size(0), -1) # last CNN faltten con. Linear NN
        out = self.fc1(out) # Linear function (readout)
        out = self.fc2(out)
        out = self.output(out)

        return out

model = CNN_Model()
from torchsummary import summary
summary(model.to(DEVICE), (3, 227, 227))

##################
# Train
#################
#from tqdm import tqdm_notebook as tqdm
# show train sechdule line
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARN_RATE, momentum = 0.9)

valid_loss_min = np.Inf # track change in validation loss
train_losses,valid_losses=[],[]

for epoch in range(1, EPOCH + 1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('running epoch: {}'.format(epoch))
    
    ############train############
    model.train()
    #for data, target in train_loader:
    for batch_idx, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        data, target = data.to(DEVICE), target.to(DEVICE)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)

        
    ######### valid##########
    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        # move tensors to GPU if CUDA is available
        data, target = data.to(DEVICE), target.to(DEVICE)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_losses.append(train_loss/len(train_loader.dataset))
    valid_losses.append(valid_loss.item()/len(valid_loader.dataset))
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # logging training/validation statistics 
    s = ('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,valid_loss))
        torch.save(model.state_dict(), MODEL_SAVE)
        valid_loss_min = valid_loss


#############
# Test
###############
def test(loaders, model, criterion):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders):
        # move to GPU
        data, target = data.to(DEVICE), target.to(DEVICE)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
    #logging    
    s = ('Test Loss: {:.6f} | Test Accuracy: %2d%% (%2d/%2d)'.format(
        test_loss , 100. * correct / total, correct, total))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)
    

model.to(DEVICE)
test(test_loader, model, criterion)








