import os
from typing import Tuple

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# set batch_size
BATCH_SIZE = 4

# set number of workers
NUM_WORKERS = 2
CLASSES: Tuple[str] = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
                       'horse', 'ship', 'truck')
DATA_DIR = "./data"
IMAGES_DIR = "./sample_images"


def load_data(data_dir: str = DATA_DIR) -> Tuple[torch.utils.data.DataLoader]:
    ''' loads the data '''
    # set transform
    transform = transforms.Compose(
       [
          transforms.ToTensor(), 
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # load train data
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # load test data
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    
    return trainloader, testloader


def imshow(images) -> None:
    ''' function to show image '''
    images_grid = torchvision.utils.make_grid(images)
    images_grid = images_grid / 2 + 0.5 # unnormalize
    npimg = images_grid.numpy() # convert to numpy objects
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def save_images_as_png(num_batches: int = 2):
    ''' saves images as png '''
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    _, testloader = load_data()
    dataiter = iter(testloader)

    to_pil = transforms.ToPILImage()

    for batch in range(num_batches):
        # Images tensor of shape: Size
        images_tensor, _ = next(dataiter)

        for i, image in enumerate(images_tensor):
            # Convert to PIL Image
            image = to_pil(image)
            # Save the image
            filename = os.path.join(IMAGES_DIR, f"{batch}_{i}.png")
            image.save(filename)


if __name__ == "__main__":
    save_images_as_png()