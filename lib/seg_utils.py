import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

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
    
