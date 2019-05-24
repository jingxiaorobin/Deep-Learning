import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.501, 0.473, 0.423],
                                     std=[0.226, 0.222, 0.225])

data_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

validation_data_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
