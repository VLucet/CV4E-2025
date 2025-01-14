
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MDclassDataset(Dataset):
    '''
        Constructor. Here, we collect and index the dataset inputs and
        labels.
    '''
    def __init__(self, cfg, x_df, y_df, split, model, device="cuda"):
        
        self.basepath = cfg['basepath']
        self.split = split
        self.data = x_df
        self.label = y_df
        
        if model == "resnet101":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            raise Exception("Model unknown")
        
        self.device = device

    def __getitem__(self, index):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_path = self.data.iloc[index]
        
        if self.basepath is not None:
            image_path = os.path.join(self.basepath, image_path)
          
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(
                self.label.iloc[index], dtype=torch.long
            )
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)