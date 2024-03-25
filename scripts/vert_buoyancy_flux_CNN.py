import numpy as np
import sys
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import glob
import xarray as xr

import submeso_ml.systems.regression_system as regression_system
import submeso_ml.cnn.fcnn as fcnn
import submeso_ml.data.dataset as dataset


# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


# res_string can be one of the following ['1_12','1_8','1_4','1_2','1']
# default is set to '1_4' unless perscribed otherwise

res_string = '1_4'
input_string = ['grad_B','FCOR' , 'HML', 'TAU', 'Q', 'div', 'vort', 'strain']


def vert_buoyancy_flux_CNN(input_string=input_string, res_string=res_string):
    """ Compute vertical buoyancy flux with pre-trained CNN following Bodner et al (2024) """
    if Is_None(db,H):
        return None
    else:
        # perscribe input features corresponding to pre-trained model. Note that this model does not include boudnarhy layer depth 'HBL'
        submeso_dataset=dataset.SubmesoDataset(input_string,res=res_string)
        
        #define test loader, which is the batch being passed in CNN. 
        test_loader=DataLoader(
            submeso_dataset,
            batch_size=len(submeso_dataset.test_ind),
            sampler=submeso_dataset.test_ind)
        
        
        # load pre-trained model without HBL  ******* CHANGE PATH *******
        model = torch.load('NEMO_MLE/trained_models/fcnn_k5_l7_m_HBL_res_'+res_strg[i_res]+'.pt')
        
        # passing the entire batch in test_loader into the CNN to get prediction of w'b'                
        for x_data in test_loader:
            prediction = model(x_data.to(device)).detach().numpy() 
        
        return prediction
