import os
import torch
import numpy as np
import json
import pickle as pkl
from PIL import Image

"""
This dataloader assumes that you have
    1. sampled 12228 points from all datacloud
    2. generated cld_rgb_nrm from the sampled point cloud
    3. saved the [label, index] for the sampled point cloud

"""
scene_id_leading_zeros = 0
img_id_leading_zeros = 0
rgb_format = '.png'
dpt_format = '.png'
msk_format = '.png'
seg_format = '.npy'


class Dataset():

    def __init__(self, root_path, scene_id):
        
        #self.dataset_name = dataset_name
        self.root_dir = root_path
        self.dat_dir = root_path + "/" + scene_id
        self.dir = self.dat_dir
        self.rgb_dir = self.dir +'/rgb' 

            
    def get_item(self, idx):

        try:
            with Image.open(os.path.join(self.rgb_dir,str(idx).zfill(img_id_leading_zeros)+ rgb_format)) as ri:
                rgb = np.array(ri)[:, :, :3]
                rgb = np.transpose(rgb, (2, 0, 1))

            with open(os.path.join(self.dir, 'cld_rgb_nrms/{}.pkl'.format(idx)),'rb') as f:
                    cld_rgb_nrms = pkl.load(f)

            label_data = np.load(os.path.join(self.dir, 'labels/{}.npy'.format(idx)), allow_pickle=True)
            
            labels = label_data[0]
            choose = label_data[1]
            cld_rgb_nrms = np.asarray(cld_rgb_nrms)

        except:
            print("Error occured while loading data")     
        
        return torch.LongTensor(labels.astype(np.int32)),\
               torch.from_numpy(cld_rgb_nrms.astype(np.float32)), torch.LongTensor(choose.astype(np.int32)),\
               torch.from_numpy(rgb.astype(np.float32))
    
    def __len__(self):
        return len(os.listdir(self.rgb_dir))
    
    def __getitem__(self, idx):
        data = self.get_item(idx)
        return data