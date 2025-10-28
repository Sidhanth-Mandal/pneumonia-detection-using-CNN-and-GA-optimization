from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class PneumoniaDataset(Dataset):
    """Custom dataset for chest X-ray images"""

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load NORMAL images (label 0)
        normal_dir = self.root_dir / 'NORMAL'
        if normal_dir.exists():
            for img_path in normal_dir.glob('*.jpeg'):
                self.images.append(str(img_path))
                self.labels.append(0)
        
        # Load PNEUMONIA images (label 1)
        pneumonia_dir = self.root_dir / 'PNEUMONIA'
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob('*.jpeg'):
                self.images.append(str(img_path))
                self.labels.append(1)
        
        print(f"{split} set: {len(self.images)} images ({sum(self.labels)} pneumonia, {len(self.labels) - sum(self.labels)} normal)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label