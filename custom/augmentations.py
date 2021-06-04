import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


def to_tensor(p: int = 1.0):
    return ToTensorV2(p=p)


class SaltAndPepperNoise(ImageOnlyTransform):
    def __init__(self, s_vs_p: float, amount: float, always_apply: bool = False, p: float = 0.5):
        super(SaltAndPepperNoise, self).__init__(always_apply, p)

        self.s_vs_p = s_vs_p
        self.amount = amount

    def apply(self, img, **params):
        salt = np.ceil(self.amount * img.size * self.s_vs_p)
        salt_coordinates = [np.random.randint(0, i - 1, int(salt)) for i in img.shape]

        img[tuple(salt_coordinates)] = 1

        return img
