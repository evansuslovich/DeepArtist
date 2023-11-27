import torch
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageFolder


ROOT = './DATA'


transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((224, 224), antialias=True),  # Explicitly set antialias to True
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust if images are grayscale
])


def load(root: str) -> None:

    dataset = ImageFolder(root=root, transform=transform)





if __name__ == '__main__':
    
    dataset = load(ROOT)
    print(dataset)
