

import os
from tkinter import Label

from numpy import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#os.environ["MKL_NUM_THREADS"] = "12"
#os.environ["NUMEXRR_NUM_THREADS"] = "12"
#os.environ["OMP_NUM_THREADS"] = "12"
#os.environ["openmp"] = "True"

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import DataLoader
import time
import matplotlib.image as pm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch.utils.data as Data
import scipy.io as sio
import scipy
from scipy import io
#import net1x1_param
from Denese_simple import net1x1
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.parallel
#from datasets import inverse_data_transform
import h5py 
import glob 
import torch.optim as optim
import torch.optim
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def getData_fastMRI(nt=50):
    print('Reading data')
    tic()
    #data_path = './Train_data/*.mat'
    #data_dir = glob.glob(data_path)
    for i in range(1):


        #Load Reg Data
        label_data_dir = './Test_Gen_data/Test_Label_4D_reg.mat'
        Label_data = scipy.io.loadmat(label_data_dir)['Reg']
        Rec_data_dir = './Test_Gen_data/Test_Rec_4D_reg.mat'

        Rec_data = scipy.io.loadmat(Rec_data_dir)['Reg']
        print('Label_data:',Label_data.shape)
        
        #1 2
        TSL_data_1 = Rec_data[...,0:1]
        TSL_data_5 = Rec_data[...,4:5]

        TSL_data_2 = Label_data[...,1:2]
        TSL_data_3 = Label_data[...,2:3]
        TSL_data_4 = Label_data[...,3:4]        
        
        # #3 4
        # TSL_data_1 = Rec_data[...,0:1]
        # TSL_data_5 = Rec_data[...,4:5]

        # TSL_data_2 = Label_data[...,1:2]
        # TSL_data_3 = Label_data[...,2:3]
        # TSL_data_4 = Label_data[...,3:4]

        TSL_data_label_all = np.concatenate((TSL_data_2,TSL_data_3,TSL_data_4),axis = -1)
        TSL_data_rec_all = np.concatenate((TSL_data_1,TSL_data_5),axis = -1)
    TSL_data_label_all = np.transpose(TSL_data_label_all,(0,3,1,2))
    TSL_data_rec_all = np.transpose(TSL_data_rec_all,(0,3,1,2))
    print('TSL_data_rec_all:',TSL_data_rec_all.shape)

    TSL_label = abs(TSL_data_label_all)
    TSL_rec = abs(TSL_data_rec_all)
    mask = abs(TSL_data_rec_all)
    return TSL_label,TSL_rec,mask
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#org, atb, mask= getData_brain(nImg=360)
org, atb, mask = getData_fastMRI(nt=50)
org = torch.from_numpy(org)
atb = torch.from_numpy(atb)
mask = torch.from_numpy(mask)
nb, nc, nx, ny = org.shape
class Mydata(Data.Dataset):
    def __init__(self):
        super(Mydata,self).__init__()
        self.org = org
        self.atb = atb
        self.mask = mask
    def __getitem__(self,index):
        orgk = self.org[index]
        atbk = self.atb[index]
        mask = self.mask[index]
        return orgk, atbk, mask
    def __len__(self):
        return len(self.org)

batch_size = 1
lr =0.0001
start_epoch =0
n_epochs = 1
trainset = Mydata()
train_loader = Data.DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=False)
print('Train_loader:',len(train_loader))
# define model
#G_model = net1x1_param.net1x1().cuda()
G_model = net1x1().cuda()

# define loss function
# criterion = nn.MSELoss(size_average=True, reduce=True).cuda()
loss_fun = nn.MSELoss('none').to(device)
optimizer = optim.Adam(G_model.parameters(),lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
num_samples = len(train_loader)
num_samples = 86
log_dir_1 = './model_reg/Gen_model.pth'
#log_dir_1 = './model_reg_label/Gen_model.pth'

for epoch in range(start_epoch,n_epochs):
    print('****'*20)
    print('epoch{}'.format(epoch))
    print('****'*20)
    G_model.load_state_dict(torch.load(log_dir_1))
    G_model.eval()
    #scheduler.step()
    for i, (org,atb,mask) in enumerate(train_loader):
        label_TSL,input_TSL,mask =  org.type(torch.FloatTensor).to(device),atb.type(torch.FloatTensor).to(device),mask.to(device)
        input_TSL = input_TSL/torch.max(label_TSL)
        label_TSL = label_TSL/torch.max(label_TSL)

        #Train
        #optimizer.zero_grad()
        Gen_TSL = G_model(input_TSL)
        loss = loss_fun(Gen_TSL, label_TSL)
        Gen_TSL_np = Gen_TSL.cpu().detach().numpy()
        Label_TSL_np = label_TSL.cpu().detach().numpy()
        #Input TSL
        Input_TSL_np = input_TSL.cpu().detach().numpy()
       
        Gen_TSL_np = np.transpose(Gen_TSL_np)
        Label_TSL_np = np.transpose(Label_TSL_np)
        Input_TSL_np = np.transpose(Input_TSL_np)


        #print('Gen_TSL_np:',Gen_TSL_np.shape)
        # print("Epoch:{}/{} Batch:{}/{} Loss:{:.6f} Epoch Loss:{:.6f} Lr:{}".format(epoch, n_epochs, i + 1,
        #                                                                                  round(num_samples / batch_size),
        #                                                                                  loss,
        #                                                                                  scheduler.get_lr()[0]))
        print("Epoch:{}/{} Batch:{}/{} Loss:{:.6f}".format(epoch, n_epochs, i + 1,round(num_samples / batch_size),loss))
        if i == 0:
            Gen_TSL_np_all = Gen_TSL_np
            Label_TSL_np_all = Label_TSL_np
            Input_TSL_np_all = Input_TSL_np
        else:
            Gen_TSL_np_all = np.concatenate((Gen_TSL_np_all,Gen_TSL_np),axis=-1)
            Label_TSL_np_all = np.concatenate((Label_TSL_np_all,Label_TSL_np),axis=-1)
            Input_TSL_np_all = np.concatenate((Input_TSL_np_all,Input_TSL_np),axis=-1)

    sio.savemat('./Gen_Result/Gen_data_reg.mat', {'output': Gen_TSL_np_all})
    sio.savemat('./Gen_Result/Label_data_reg.mat', {'label': Label_TSL_np_all})
    sio.savemat('./Gen_Result/Input_data_reg.mat', {'label': Input_TSL_np_all})


    print('result saved')
    print('**********************G-net Finish!!!*****************')
