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
        self.dir = root_path + "/" + scene_id
        self.rgb_dir = self.dir +'/rgb'
        self.dpt_dir = self.dir+'/depth'
        self.segmap_dir = self.dir+'/seg_maps'
        self.n_sample_points = 8192 + 4096   #12288
            
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
    
    def get_cam_info(self,idx):
                scene_cam_path = os.path.join(self.dir,'scene_camera.json')
                if os.path.exists(scene_cam_path): 
                    with open(scene_cam_path,"r") as k:
                        for i,j in enumerate(k):
                            im_dict = json.loads(j)
                            if i == idx:
                                this_cam = im_dict
                                
                        cam_K = this_cam[str(idx)]['cam_K']
                        dpt_cam_K = this_cam[str(idx)]['dpt_cam_K']
                        K = np.array(cam_K).reshape(3,3)
                        dpt_K = np.array(dpt_cam_K).reshape(3,3)
                        cam_scale =  this_cam[str(idx)]['depth_scale']

                    return K,dpt_K, cam_scale

                else:
                    print("missing scene_camera.json :")
                    
    
    def __getitem__(self, idx):
        data = self.get_item(idx)
        return data