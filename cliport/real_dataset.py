from torch.utils.data import Dataset, DataLoader 
import os
import cv2
import numpy as np
from icecream import ic
from torchvision import transforms
import torch


class RealCamDataset(Dataset):
    """
    Dataset class for the real camera data.

    params:
        data_dir: the directory of the real camera images (RGB+Depth)
    """

    def __init__(self, data_dir):
        
        self.shape = (160, 320)
        self.in_shape = (160, 320, 6)
        
        # read images in the file
        self.depth_data = []
        self.rbg_data = []
        
        for file in os.listdir(data_dir):
            if file.endswith('rgb.jpeg'):
                self.rbg_data.append(os.path.join(data_dir, file))
                self.depth_data.append(os.path.join(data_dir,file.replace('rgb.jpeg', 'depth.jpeg')))


    def transform(self, img):
        """
        transform the image into tensor
        """

        img = transforms.ToTensor()(img)
        img = torch.transpose(img, 0, 2)
        
        return img


    def get_input_tensor(self, color, depth):
        """
        propocess the image into the format that the model can take
        params:
            obs: dict with keys 'color' and 'depth' with shape [320, 160, 3] and [320, 160] respectively
        """
        cmap, hmap = color, depth
        img = np.concatenate((cmap,
                              hmap[Ellipsis, None], # [320, 160, 1]
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape # [320 160 6]
        return img

    def __len__(self):
        return len(self.rbg_data)
    
    def __getitem__(self, index):

        raw_color = cv2.imread(self.rbg_data[index])
        raw_depth = cv2.imread(self.depth_data[index], cv2.IMREAD_GRAYSCALE)

        
        h = raw_color.shape[0]
        w = raw_color.shape[1]

        # reshape image to 1/2
        raw_color = cv2.resize(raw_color, (int(w/2), int(h/2)))
        raw_depth = cv2.resize(raw_depth, (int(w/2), int(h/2)))

        h = raw_color.shape[0]
        w = raw_color.shape[1]
        
        # crop the image in the middle point
        h_new, w_new = self.shape
        color = raw_color[int(h/2 - h_new/2): int(h/2 + h_new/2), int(w/2 - w_new/2): int(w/2 + w_new/2), :]
        depth = raw_depth[int(h/2 - h_new/2): int(h/2 + h_new/2), int(w/2 - w_new/2): int(w/2 + w_new/2)]

        img = np.concatenate((color,
                              depth[Ellipsis, None], # [320, 160, 1]))
                              depth[Ellipsis, None],
                              depth[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape # [320 160 6]
        # normalize the color and dept

        return self.transform(img)


# test the daaset with main

if __name__ == '__main__':

    real_cam_dataset = RealCamDataset('real_cam_data')
    real_cam_dataloader = DataLoader(real_cam_dataset, batch_size=1, shuffle=False)
    print(len(real_cam_dataloader))

    for batch in real_cam_dataloader:
        
        
        break
    

