from common import GroundTruthDataSets, NoGroundTruthDataSets
from eval import load_data
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

TO_TENSOR = ToTensor()

class NoGroundTruthImageDataset(Dataset):
    def __init__(
        self,
        dataset_name: NoGroundTruthDataSets,
    ):
        self.data = load_data()[dataset_name]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        image_pair = self.data[idx]

        dark_image: Image = image_pair["low_image"]

        return TO_TENSOR(dark_image)


class GroundTruthImageDataset(Dataset):
    def __init__(
        self,
        dataset_name: GroundTruthDataSets,
    ):
        self.data = load_data()[dataset_name]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_pair = self.data[idx]

        dark_image: Image = image_pair["low_image"]
        bright_image: Image = image_pair["high_image"]

        return TO_TENSOR(dark_image), TO_TENSOR(bright_image)
