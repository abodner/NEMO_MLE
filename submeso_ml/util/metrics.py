import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def get_offline_metrics(model,valid_loader):
    ''' Get model predictions over entire validation set. Calculate R2, and correlation
        Returns: r2, corr
        add here per region '''

    for data in enumerate(valid_loader):
        x_data, y_data = data

        #r2
        r2 = r2_score(model(x_data).detach().cpu().numpy(), y_data.detach().cpu().numpy())

        #correlation coefficient
        corr, _ = pearsonr(model(x_data).detach().cpu().numpy().flatten(), y_data.detach().cpu().numpy().flatten())

    return r2, corr
