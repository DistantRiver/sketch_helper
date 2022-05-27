import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
from model import SHNet
from train_model import train_model
from load_data import load_data
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    MAX_EPOCH = 1
    batch_size = 4
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0
    hyper_parameters = {'alpha': 0.0003}

    print('...Data loading is beginning...')
    
    data_loader = load_data(txt_paths=['../train_stroke.txt', '../valid_stroke.txt', '../test_stroke.txt'], batch_size=batch_size)

    print('...Data loading is completed...')
    
    model_ft = SHNet().to(device)
    
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas, weight_decay=weight_decay)
    # scheduler = None
    # optimizer = optim.SGD(params_to_update, lr=0.0002, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,400], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-6)

    print('...Training is beginning...')
    # Train and evaluate
    model_ft, loss_hist = train_model(model_ft, data_loader, hyper_parameters, optimizer, scheduler, device, MAX_EPOCH, 'shnet.pth')
    
    print('...Training is completed...')

