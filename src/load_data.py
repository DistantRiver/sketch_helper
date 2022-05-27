from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import cv2

input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

class SHDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.labels = [int(path.rsplit(os.sep, 2)[1]) for path in paths]

    def __getitem__(self, index):
        path = self.paths[index]
        imgs = []
        labels = []
        label = self.labels[index]
        for i in range(1, 5): # 根据论文，LSTM不使用第一张图
            img_path = path + '_' + str(i) + '.png'
            img = cv2.imread(img_path)
            img = input_transform(img).numpy()
            imgs.append(img)
            labels.append(label)
        imgs = np.asarray(imgs)
        labels = np.asarray(labels)
        return imgs, labels

    def __len__(self):
        count = len(self.paths)
        assert len(self.paths) == len(self.labels)
        return count

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        txt_list = f.readlines()
    txt_list = [item.strip() for item in txt_list]
    return txt_list
    
def load_data(txt_paths, batch_size, phase='train'):
    train_txt_path, valid_txt_path, test_txt_path = txt_paths
    train_paths = read_txt(train_txt_path)
    valid_paths = read_txt(valid_txt_path)
    test_paths = read_txt(test_txt_path)
    
    if phase == 'train':
        paths = {'train': train_paths, 'valid': valid_paths}
        dataset = {x: SHDataset(paths[x]) for x in ['train', 'valid']}
    
        shuffle = {'train': True, 'valid': False}
    
        dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'valid']}
        
    else:
        paths = {'test': test_paths}
        dataset = {x: SHDataset(paths[x]) for x in ['test']}
    
        shuffle = {'test': False}
    
        dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0) for x in ['test']}
    
    return dataloader
