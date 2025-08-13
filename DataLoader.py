import torch,os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
class Camelyondataset(Dataset):
    def __init__(self, train_df, num_classes = 1):
        
        self.train_df = train_df
        self.num_classes = num_classes

        print('Loading ***********: len {}'.format(len(train_df)))
       
    def __getitem__(self, item):
        # get image id
        csv_file_df = self.train_df.iloc[item]
        
        feats_csv_path = csv_file_df.iloc[0]
        
        feats_csv_path = os.path.join("FEATURES_DIRECTORY/Camelyon16/c16_res50_1024", feats_csv_path)
        #feats_csv_path = os.path.join("FEATURES_DIRECTORY/Camelyon16/c16_plip_512", feats_csv_path)
        
        feats = torch.load(feats_csv_path.replace("csv", "pt"),weights_only=True)
        label = np.zeros(self.num_classes)
        
        if self.num_classes==1:
            label[0] =  csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1])<=(len(label)-1):
                label[int(csv_file_df.iloc[1])] = 1

        return label, feats

    def __len__(self):
        return len(self.train_df)


class TCGAdataset(Dataset):
    def __init__(self, train_df, num_classes = 1):
        
        self.train_df = train_df
        self.num_classes = num_classes

        print('Loading ***********: len {}'.format(len(train_df)))
       
    def __getitem__(self, item):
        # get image id
        csv_file_df = self.train_df.iloc[item]

        feats_csv_path = csv_file_df.iloc[0]

        feats_csv_path = os.path.join("FEATURES_DIRECTORY/tcga_nsclc/feature/20x_res50_1024", feats_csv_path)
        #feats_csv_path = os.path.join("FEATURES_DIRECTORY/tcga_nsclc/feature/20x_plip_512", feats_csv_path)
        
        feats = torch.load(feats_csv_path.replace("csv", "pt"),weights_only=True)
        label = np.zeros(self.num_classes)
        
        if self.num_classes==1:
            label[0] =  csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1])<=(len(label)-1):
                label[int(csv_file_df.iloc[1])] = 1

        return label, feats

    def __len__(self):
        return len(self.train_df)