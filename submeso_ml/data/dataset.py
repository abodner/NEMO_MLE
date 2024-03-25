from torch.utils.data import Dataset
import numpy as np
import random
import torch


class SubmesoDataset(Dataset):
    def __init__(self,input_features = ['grad_B','FCOR', 'HML', 'TAU',
              'Q', 'HBL', 'div', 'vort', 'strain'],res='1_4',loc_num=0,seed=123,train_split=0.8):
        super().__init__()
        self.seed=seed
        self.train_split=train_split
        self.input_features=input_features
        self.path= '/scratch/ab10313/pleiades/NN_data_'+'%s' % res+'/'
        self.loc = loc_num

        # load features for input
        in_features=[]
        for input_feature in self.input_features:
            in_features.append(np.load(self.path+'%s.npy' % input_feature))
        # x input
        self.x = torch.from_numpy(np.stack(in_features,axis=1)).float()

        # load output
        WB_sg = np.load(self.path+'WB_sg.npy')

        # y output
        self.y = torch.from_numpy(np.tile(WB_sg,(1,1,1,1)).reshape(WB_sg.shape[0],1,WB_sg.shape[1],WB_sg.shape[2])).float()

        self._get_split_indices()
        self._norm_factors()
        self._norm_data()

    def _get_split_indices(self):
        """ obtain a set of train and test indices """

        # randomnly generate train, test and validation time indecies
        time_ind = self.x.shape[0]
        rand_ind = np.arange(time_ind)
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(rand_ind)
        self.train_ind, self.test_ind =  rand_ind[:round(self.train_split*time_ind)], rand_ind[round((self.train_split)*time_ind):]

        # sort test_ind
        self.train_ind = np.sort(self.train_ind)


    def _norm_factors(self):
        """ load global noramlization factors: mean and std """
        self.y_mean = torch.from_numpy(np.load(self.path+'WB_sg_mean.npy')*np.ones(self.y.shape)).float()
        self.y_std = torch.from_numpy(np.load(self.path+'WB_sg_std.npy')*np.ones(self.y.shape)).float()

        # load mean and std from features for input
        std_in_features=[]
        mean_in_features=[]
        for input_feature in self.input_features:
            mean_in_features.append(np.load(self.path+'%s_mean.npy' % input_feature)*np.ones((self.x.shape[0],self.x.shape[2],self.x.shape[3])))
            std_in_features.append(np.load(self.path+'%s_std.npy' % input_feature)*np.ones((self.x.shape[0],self.x.shape[2],self.x.shape[3])))

        self.x_mean = torch.from_numpy(np.stack(mean_in_features,axis=1)).float()
        self.x_std = torch.from_numpy(np.stack(std_in_features,axis=1)).float()


    def _norm_data(self):
        """ normalize inputs and output to global normalization factors"""
        self.x_norm = (self.x - self.x_mean)/self.x_std
        self.y_norm = (self.y - self.y_mean)/self.y_std


    def __getitem__(self,idx):
        return (self.x_norm[idx])



