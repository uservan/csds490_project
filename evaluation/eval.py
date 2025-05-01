import __init__
from typing import Any, Callable
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from eval_with_ground import evaluate_image_pair_from_pil
from eval_without_ground import evaluate_image
import numpy as np
import cv2
from cv2.typing import MatLike
from dehaze import dehaze_and_enhance
from PIL import Image
import re


def load_data() -> defaultdict[str, list[dict[str, Any]]]:
    dataset = load_dataset("VanWang/low_datasets")
    pair_dict = defaultdict(lambda: defaultdict(dict))

    for sample in tqdm(dataset["train"]):
        source = sample["source"]
        label = sample["label"]
        # 只取 name 中的数字
        name = re.findall(r"\d+", sample["name"])
        name = name[0] if name else sample["name"]  # 若没有数字，保留原始 name

        pair_dict[source][name][label] = sample["image"]

    paired_samples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for source, name_dict in pair_dict.items():
        for name, img_dict in name_dict.items():
            if "low" in img_dict and "high" in img_dict:
                paired_samples[source].append(
                    {
                        "source": source,
                        "name": name,
                        "low_image": img_dict["low"],
                        "high_image": img_dict["high"],
                    }
                )
            else:
                paired_samples[source].append(
                    {
                        "source": source,
                        "name": name,
                        "low_image": img_dict.get("low"),
                    }
                )

    return paired_samples


# Convert PIL Image to OpenCV format
# PIL → OpenCV (RGB → BGR)
def pil_to_cv2(pil_image: Image.Image) -> MatLike:
    rgb = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


# OpenCV → PIL (BGR → RGB)
def cv2_to_pil(cv_img: MatLike) -> Image.Image:
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def eval(method: Callable[[MatLike], MatLike] = dehaze_and_enhance):
    """
    Evaluate the dehazing and enhancement method on the dataset.
    Args:
        method: input image is opencv2 format, output image is opencv2 format
    Returns:
        A dictionary containing the evaluation results for each dataset.
    """
    paired_samples = load_data()
    all_results = dict()
    for dataset in paired_samples.keys():
        samples = paired_samples[dataset]
        results = defaultdict(list)
        for sample in tqdm(samples):
            low_image = sample["low_image"]
            high_image = sample.get("high_image")
            reference_img = cv2_to_pil(method(pil_to_cv2(low_image)))
            # reference_img.show()
            if high_image is not None:
                scores = evaluate_image_pair_from_pil(reference_img, high_image)
                # Compute metrics here
            else:
                scores = evaluate_image(reference_img, low_image)
            for key, s in scores.items():
                results[key].append(s)
        all_results[dataset] = results
    return all_results
