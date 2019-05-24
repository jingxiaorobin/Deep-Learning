import torchvision.transforms as transforms


data_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ColorJitter(0.8, contrast=0.3),
    transforms.RandomAffine(10, scale=(0.8, 1.2), translate=(0.2, 0.2)),
    transforms.RandomHorizontalFlip(),            #flip transform
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

validation_data_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
