import torchvision.transforms as transforms


data_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ColorJitter(0.8, contrast=0.3),
    transforms.RandomAffine(20, scale=(0.8, 1.2), translate=(0.2, 0.2)),
    transforms.RandomHorizontalFlip(),            #flip transform
    transforms.ToTensor(),
    transforms.Normalize((0.50322117, 0.47465022, 0.42753401), (0.22680563, 0.22247544, 0.22557994))
])

validation_data_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.50322117, 0.47465022, 0.42753401), (0.22680563, 0.22247544, 0.22557994))
])
