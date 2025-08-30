from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def build_transforms(img_size: int, train: bool):
    if train:
        aug = [
            A.Resize(img_size + 16, img_size + 16),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.ColorJitter(0.1, 0.1, 0.1, 0.05, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ]
    else:
        aug = [
            A.Resize(img_size, img_size),
            A.CenterCrop(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    return A.Compose(aug)


class AlbumentationsFolder(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root)
        self.aug = transform

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.aug(image=image)["image"]
        return image, target

def make_loaders(train_dir, val_dir, img_size, batch_size, num_workers):
    t_train = build_transforms(img_size, True)
    t_val = build_transforms(img_size, False)
    ds_train = AlbumentationsFolder(train_dir, t_train)
    ds_val = AlbumentationsFolder(val_dir, t_val)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, ds_train.classes
