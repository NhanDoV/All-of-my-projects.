import os
import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
from SeResNet import *
import warnings 
warnings.filterwarnings('ignore')
#========================================================
input_data_dir = '../input/prostate-cancer-grade-assessment'
files = os.listdir(input_data_dir)
train = pd.read_csv(f'{input_data_dir}/train.csv')
test = pd.read_csv(f'{input_data_dir}/test.csv')
sample = pd.read_csv(f'{input_data_dir}/sample_submission.csv')
#========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#========================================================
seed_torch(seed=53) 
#========================================================
train_dataset = TrainDataset(train, train[CFG.target_col], transform=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
#===================== Train test split =========================================
if CFG.debug:
    folds = train.sample(n=2000, random_state=CFG.seed).reset_index(drop=True).copy()
else:
    folds = train.copy()
#========================================================
train_labels = folds[CFG.target_col].values
kf = StratifiedKFold(n_splits=CFG.n_fold, 
                     shuffle=True, 
                     random_state=CFG.seed)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
    folds.loc[val_index, 'fold'] = int(fold)
folds['fold'] = folds['fold'].astype(int)
folds.to_csv('folds.csv', index=None)
pretrained_path = {...}
#========================================================
preds = []
valid_labels = []
for fold in range(CFG.n_fold):
    _preds, _valid_labels = train_fn(folds, device, fold)
    preds.append(_preds)
    valid_labels.append(_valid_labels)
#========================================================
preds = np.concatenate(preds)
valid_labels = np.concatenate(valid_labels)
#========================================================
optimized_rounder = OptimizedRounder()
optimized_rounder.fit(preds, valid_labels)
coefficients = optimized_rounder.coefficients()
final_preds = optimized_rounder.predict(preds, coefficients)
submission = submit(device, sample, coefficients, dir_name='test_images')
submission['isup_grade'] = submission['isup_grade'].astype(int)
submission.to_csv('submission.csv', index=False)
#========================================================