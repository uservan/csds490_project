from typing import Literal

from cv2.typing import MatLike
from eval import load_data, pil_to_cv2
from torch.utils.data import Dataset


class NoGroundTruthImageDataset(Dataset):
    def __init__(
        self,
        dataset_name: Literal["Dark_Face"],
    ):
        self.data = load_data()[dataset_name]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MatLike:
        image_pair = self.data[idx]

        dark_image = pil_to_cv2(image_pair["low"])

        return dark_image


class GroundTruthImageDataset(Dataset):
    def __init__(
        self,
        dataset_name: Literal["lol_dataset", "LOL-V2"],
    ):
        self.data = load_data()[dataset_name]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[MatLike, MatLike]:
        image_pair = self.data[idx]

        dark_image = pil_to_cv2(image_pair["low"])
        bright_image = pil_to_cv2(image_pair["high"])

        return dark_image, bright_image
