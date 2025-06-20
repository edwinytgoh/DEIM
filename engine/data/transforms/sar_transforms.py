import torch
import torchvision.transforms as T
from torchvision.tv_tensors import Image, Mask

from ...core import register


@register()
class SARSafeNormalize(T.Normalize):
    # """Mean/std on valid pixels only (nodata == 0)."""
    # def _transform(self, x, _) -> torch.Tensor:
    #     mask = x[0] != 0
    #     y = x.clone()
    #     m = torch.as_tensor(self.mean, dtype=x.dtype,
    #                         device=x.device)[:, None, None]
    #     s = torch.as_tensor(self.std, dtype=x.dtype,
    #                         device=x.device)[:, None, None]
    #     y[:, mask] = (x[:, mask] - m) / s
    #     return y
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[..., None, None]
        self.std = torch.tensor(std)[..., None, None]

    # DEIM transform signature: img, target, dataset -> returns same triple
    def __call__(self, image, target=None, _=None):
        torch_mode = isinstance(image, torch.Tensor)
        img = image.clone() if torch_mode else np.asarray(image).copy()

        # nodata mask (any channel = 0 → treat as nodata)
        mask = (img[..., 0] == 0) if torch_mode else (img[..., 0] == 0.0)

        img = (img - self.mean) / self.std
        img[mask] = 0.0

        return (img, target, _) if target is not None else img
