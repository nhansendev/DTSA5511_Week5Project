import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


from constants import RNG


def make_transforms(resize=286, crop=256):
    transform_list = []
    transform_list.append(
        transforms.Resize([resize, resize], transforms.InterpolationMode.BICUBIC)
    )
    transform_list.append(transforms.RandomCrop(crop))
    transform_list.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(transform_list)


class CustomDataset(Dataset):
    def __init__(self, dataA, dataB, preallocate=False) -> None:
        super().__init__()
        self.preallocate = preallocate

        self.dataA = torch.permute(
            torch.FloatTensor(dataA) / 127.5 - 1, (0, 3, 1, 2)
        ).to("cuda" if preallocate else "cpu")
        self.dataB = torch.permute(
            torch.FloatTensor(dataB) / 127.5 - 1, (0, 3, 1, 2)
        ).to("cuda" if preallocate else "cpu")

        self.transform = make_transforms().cuda()

        self.N_A = len(self.dataA)
        self.N_B = len(self.dataB)

    def __len__(self):
        return len(self.dataA)

    def __getitem__(self, index):
        return (
            self.transform(self.dataA[index % self.N_A].cuda()),
            self.transform(self.dataB[index % self.N_B].cuda()),
        )


class ImagePool:
    def __init__(self, size=50) -> None:
        self.size = size
        self.qty_filled = 0
        self.images = []

    def query(self, images):
        if self.size < 1:
            return images

        N = images.shape[0]
        if self.qty_filled < self.size:
            # Fill the pool first
            N = min(N, self.size - self.qty_filled)
            for i in images[:N]:
                # TODO: Need .clone()?
                self.images.append(i.clone())
            self.qty_filled += N
            return images

        # Randomly swap images
        for i, img in enumerate(images):
            if RNG.uniform() > 0.5:
                idx = RNG.integers(0, self.size)
                tmp = img.clone()
                images[i] = self.images[idx]
                self.images[idx] = tmp

        return images

    def __call__(self, images):
        return self.query(images)
