import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
from torchvision import transforms

from datetime import datetime
import torch.optim as optim
from model import SHNet
from train_model import train_model
from load_data import load_data
import os
import cv2
import pickle
import numpy as np
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters

    print('...Data loading is beginning...')
    
    data_dir = '../imgData'
    cls_num = 345
    str_num = 5
    img_num = 25000
    save_dir = '../saveData'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    print('...Data loading is completed...')
    
    model = SHNet().to(device)
    model.load_state_dict(torch.load('../weights/shnet.pth'))
    model.eval()
    
    for i in range(str_num):
        for j in range(cls_num):
            save_path = os.path.join(save_dir, str(i) + "_" + str(j) + ".list")
            print(save_path)
            with open(save_path, 'wb') as sf:
                feas = []
                for k in range(img_num):
                    img_path = os.path.join(data_dir, str(j), str(k) + '_' + str(i) + '.png')
                    img = cv2.imread(img_path)
                    img = input_transform(img).unsqueeze(0).to(device)
                    fea, str_out, cls_out = model(img)
                    fea = fea.squeeze().data.cpu().numpy()
                    fea = fea > 0
                    feas.append(fea)
                feas = np.asarray(feas)
                pickle.dump(feas, sf)

