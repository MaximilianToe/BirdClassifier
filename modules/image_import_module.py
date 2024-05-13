import torch
import os
from torch.utils.data import Dataset
import torchvision 

'''
creates a list of subdirectories in the root directory.
Each subdirectory only contains spectrograms for a single bird species and the name of the subdirectory is used to identify that species.
'''
def get_subdirectories(root_dir: str) -> list[str]:
    return [x[0] for x in os.walk(root_dir)][1:]


'''
Image dataset class that loads images from a directory and assigns a label to each image.
To assign a label to each image, the name_to_idx dictionary is used to map the name of the subdirectory to an index.
'''

class ImageDataset(Dataset):
    def __init__(self, name_to_idx, root_dir: str, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)
        self.label = root_dir.split('/')[-1]
        self.name_to_idx = name_to_idx

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        image = torchvision.io.read_image(file_path).float() 
        image = image
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.name_to_idx[self.label], dtype=torch.long)
        return image, label