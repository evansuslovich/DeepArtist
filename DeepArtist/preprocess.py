import torch
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Class definition courtesy of Prof. Venkat
class CropToSmallerDimension(object):
    
    def __init__(self) -> None:
        super().__init__()


    def __call__(self, img):

        # Get the original image size
        width, height = img.size
        
        # Determine the smaller dimension
        smaller_dimension = min(width, height)
        
        # Crop the image to the smaller dimension
        return transforms.CenterCrop(smaller_dimension)(img)


class ImageDataManager(object):

    _data_root: str
    _square_image_size: int
    _transform: int #?
    _dataset: int #?

    def __init__(self,
                 data_root: str='./Data',
                 square_image_size: int=100) -> None:
        
        super().__init__()

        self._data_root = data_root
        self._square_image_size = square_image_size

        self._transform = transforms.Compose([
            CropToSmallerDimension(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(self._square_image_size, antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self._dataset = ImageFolder(root=self._data_root, transform=self._transform)
    

    def label_map(self):
        return self._dataset.classes
    

    def transform(self, image):
        return self._transform(image)


    def split_dataset(self,
                      train_split: float=0.8,
                      validate_split: float=0.1,
                      random_seed: int=None):
        '''
        Splits the dataset into 3 subsets: train, validate, and test. The train_split and
        validate_split portions must add up to no more than 1.0. The size of the test set will be
        1.0 - train_split - validate_split.
        '''

        if train_split + validate_split > 1.0:
            raise ValueError('The portion size of the train and validate splits must sum up to no' \
                            + ' more than 1.0.')

        dataset_all_indices = list(range(len(self._dataset)))
        train_idx, posttrain_idx = train_test_split(dataset_all_indices, 
                                                    test_size=1 - train_split,
                                                    shuffle=True,
                                                    random_state=random_seed)
        
        posttrain_dataset = Subset(self._dataset, posttrain_idx)
        posttrain_all_indices = list(range(len(posttrain_dataset)))
        validate_idx, test_idx = train_test_split(posttrain_all_indices,
                                                test_size=1 - (validate_split / (1 - train_split)),
                                                shuffle=True,
                                                random_state=random_seed)

        return (
            Subset(self._dataset, train_idx),
            Subset(posttrain_dataset, validate_idx),
            Subset(posttrain_dataset, test_idx)
        )


    def split_loaders(self,
                     train_split: float=0.8,
                     validate_split: float=0.1,
                     batch_size: int=100,
                     random_seed: int=None):
        '''
        Loads the dataset, splits it into train, validate, and test subsets, then creats loaders
        with  the given batch size. Returns a tuple containing the three subset loaders.
        '''

        train_dataset, validate_dataset, test_dataset = self.split_dataset(train_split,
                                                                           validate_split,
                                                                           random_seed)

        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
            len(train_dataset),
            DataLoader(validate_dataset, batch_size=batch_size, shuffle=False),
            len(validate_dataset),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            len(test_dataset)
        )


# Just runs a test.
if __name__ == '__main__':

    idm = ImageDataManager()

    train_loader, validate_loader, test_loader = idm.split_loaders(0.8, 0.1, 100, 1)

    print(idm.label_map())

    # Show the first image as an example.
    plt.imshow(next(iter(train_loader))[0][0].permute(1, 2, 0))
    plt.show()
