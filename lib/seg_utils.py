import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
import cv2

def plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies, epoch,fn):
    # Create subplots for loss and accuracy
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot loss
    axs[0].cla()
    axs[0].plot(train_losses, label='Train')
    axs[0].plot(val_losses, label='Validation')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    axs[0].set_xticks(range(0, len(train_losses) + 1, 5))
    axs[0].grid(True)
    axs[0].plot()

    # Plot accuracy
    axs[1].cla()
    axs[1].plot(train_accuracies, label='Train')
    axs[1].plot(val_accuracies, label='Validation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend()
    axs[1].set_xticks(range(0, len(train_accuracies) + 1, 5))
    axs[1].grid(True)
    axs[0].plot()

    plt.tight_layout()
    plt.pause(0.1)  # Pause to update the plot
    
    if (epoch+1) % 50 == 0:
        if not os.path.exists('training_plots'):
            os.makedirs('training_plots')
        filename = os.path.join("training_plots", fn+".png")
        fig.savefig(filename)
       

    plt.show()

def create_scheduler(optimizer, mode='step', step_size=30, gamma=0.1, patience=5, factor=0.1, threshold=0.001, min_lr=1e-6):
    """
    Create a learning rate scheduler.

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer to adjust the learning rate for.
        mode (str): Type of scheduler. Options: 'step', 'plateau'.
        step_size (int): Period of learning rate decay (for 'step' mode).
        gamma (float): Multiplicative factor of learning rate decay (for 'step' mode).
        patience (int): Number of epochs with no improvement after which learning rate will be reduced (for 'plateau' mode).
        factor (float): Factor by which the learning rate will be reduced (for 'plateau' mode).
        threshold (float): Threshold for measuring the new optimum (for 'plateau' mode).
        min_lr (float): Lower bound on the learning rate (for 'plateau' mode).

    Returns:
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    """
    if mode == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif mode == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience,
                                                    factor=factor, threshold=threshold,
                                                    min_lr=min_lr, verbose=True)
    else:
        raise ValueError("Unsupported scheduler mode. Please choose either 'step' or 'plateau'.")
    
    return scheduler

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, target):
        if self.alpha is not None:
            # Apply class weights based on alpha
            alpha_tensor = torch.tensor(self.alpha, device=inputs.device)
            #alpha_factor = alpha_tensor[target].unsqueeze(1)
            alpha_factor = alpha_tensor[target.view(-1)].view_as(target)
            
        else:
            alpha_factor = 1.0

        # Compute focal weight for each prediction
        focal_weight = torch.pow(1 - F.softmax(inputs, dim=1), self.gamma)
        # Compute focal loss
        ce_loss = F.cross_entropy(inputs, target, reduction='none')
        #focal_loss = alpha_factor * focal_weight.unsqueeze(1) * ce_loss.unsqueeze(1)
        #print("alpha_factor",alpha_factor.shape)
        #print("focal_weight",focal_weight.shape)
        #print("ce_loss", ce_loss.shape)


        #focal_loss = alpha_factor * focal_weight * ce_loss
        focal_loss = focal_weight * ce_loss.unsqueeze(1)

        # Apply reduction method
        if self.reduction == 'mean':
            loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(focal_loss)
        else:
            loss = focal_loss

        return loss
    
#Function to project depth to pointcloud

def dpt_2_cld( depth_frame, K,segMask = None,cam_scale=1):

    w = depth_frame.shape[1]
    h = depth_frame.shape[0]
    
    xmap = np.array([[j for i in range(w)] for j in range(h)])
    ymap = np.array([[i for i in range(w)] for j in range(h)])
    
    dpt = np.array(depth_frame, dtype=np.float32)
    dpt = dpt/1000
    
    if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
    msk_dp = dpt > -1
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)

    if len(choose) < 1:
        return None, None
        
    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    
    if segMask is not None:
             
        focus_points = np.argwhere(segMask != 0)
        focus = segMask != 0
        
        # projecting only the focus
        pt2 = dpt_mskd[focus.flatten()] / cam_scale
        pt0b= (ymap_mskd[focus.flatten()] - cam_cx) * pt2 / cam_fx
        pt1b= (xmap_mskd[focus.flatten()] - cam_cy) * pt2 / cam_fy
        focus_points = np.concatenate((pt0b, pt1b, pt2),axis=1)
               
        return focus_points , choose
    
    else :

        # projecting the cloud as a whole    
        pt2 = dpt_mskd / cam_scale
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate((pt0, pt1, pt2),axis=1)
        
        return cld , choose
    
def project_p3d(p3d, cam_scale, K):
        
        if p3d.shape[1]<4:
            p3d = p3d * cam_scale
            p2d = np.dot(p3d, K.T)
            p2d_3 = p2d[:, 2]
            p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
            p2d[:, 2] = p2d_3
            p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
            return p2d
        else:
            p3d = p3d * cam_scale
            #print(p3d.shape)
            print('xyz_rgb points projected to 2D')
            p2d = np.dot(p3d[: , 0:3], K.T)
            p2d_3 = p2d[:, 2]
            filter = np.where(p2d_3 < 1e-8)
            if filter[0].shape[0]>0:
                p2d_rgbs = p3d[filter, 3:6]
                p2d_3[filter] = 1.0
            else:
                p2d_rgbs = p3d[:, 3:6]
            p2d[:, 2] = p2d_3
            p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
            
            #print(p3d.shape)
            #print(p2d.shape)

            #return np.concatenate((p2d, p2d_rgbs), axis=1).astype(np.int32)
            return p2d

def draw_p2ds( img, p2ds, color, rad):
        h, w = img.shape[0], img.shape[1]

        for pt_2d in p2ds:
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)

            if p2ds.shape[1]>2:
                img = cv2.circle(
                    cv2.UMat(img), (pt_2d[0], pt_2d[1]), rad, (int(pt_2d[2]), int(pt_2d[3]), int(pt_2d[4])) , -1
                )
            else:
                img = cv2.circle(
                    cv2.UMat(img), (pt_2d[0], pt_2d[1]), rad, color, -1
                )
            '''
            img = cv2.circle(
                img, (pt_2d[0], pt_2d[1]), rad, color, -1
            )'''
          
        return img.get()