from count_neutrophils import read_png_into_numpy
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import TracedModel

import numpy as np
import os
from PIL import Image
import torch
from typing import List


ORTHOPEDIA_DIR = "/mnt/dss/dssfs04/pn25ke/pn25ke-dss-0001/orthopedia"

CONF_THRESH = 0.3
IOU_THRESH = 0.45
MODEL_PATH = "/mnt/dss/dsshome1/09/ge59gew2/orthopedia/yolov7_neutrophil.pt"
CONFIG_PATH = "/mnt/dss/dsshome1/09/ge59gew2/orthopedia/data/lizard.yaml"
DEVICE_ID = 0
BATCH_SIZE = 128


def read_lizard_label(path: str) -> np.ndarray:
    """
    x_center, y_center, width, height -> x1, y1, width, height
    """
    with open(path, "r") as f:
        label = np.array([list(map(float, item.replace("\n", "").split(" "))) for item in f.readlines()])[:, 1:]
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    elif len(label.shape) != 2:
        raise ValueError
    label[:, 0] -= 0.5 * label[:, 2]
    label[:, 1] -= 0.5 * label[:, 3]
    return label


def create_coco_like_json(annotations, split: str, export_dir: str) -> None:
    import glob
    import fiftyone as fo

    images_patt = os.path.join("lizard", "images", split, "*")
    dataset = fo.Dataset(name=export_dir)
    for filepath in glob.glob(images_patt):
        sample = fo.Sample(filepath=filepath)
        detections = []
        for obj in annotations[filepath]:
            label = obj["label"]
            bounding_box = obj["bbox"]
            detections.append(fo.Detection(label=label, bounding_box=bounding_box, confidence=obj["score"]))
        sample["ground_truth"] = fo.Detections(detections=detections)
        dataset.add_sample(sample)
    export_dir = export_dir
    label_field = "ground_truth"
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field,
    )


def create_gt_annotations(split: str) -> None:
    return {
        os.path.join("lizard", "images", split, f): [
            {"bbox": row.tolist(), "label": "neutrophil"}
            for row in read_lizard_label(os.path.join("lizard", "labels", split, f.replace(".jpg", ".txt")))
        ]
        for f in os.listdir(os.path.join("lizard", "images", split))
    }


class Inference:
    def __init__(self, model_path: str, config_path: str, device: torch.device) -> None:
        self.device = device
        self.model = attempt_load(model_path).eval().to(device)
        self.conf_thres = CONF_THRESH
        self.iou_thres = IOU_THRESH
    
    @torch.no_grad()
    def __call__(self, images: List[np.ndarray]):
        x = torch.from_numpy(np.stack(images)).float().to(self.device).permute(0, 3, 1, 2) / 255.
        return [
            item[:, :-1]
            for item in non_max_suppression(self.model(x, augment=False)[0], self.conf_thres, self.iou_thres)
        ]


def create_pred_annotations(split: str, image_size: int) -> None:
    from tqdm import tqdm

    MODEL_PATH = "/mnt/dss/dsshome1/09/ge59gew2/orthopedia/yolov7_neutrophil.pt"
    CONFIG_PATH = "/mnt/dss/dsshome1/09/ge59gew2/orthopedia/data/lizard.yaml"
    inference = Inference(MODEL_PATH, CONFIG_PATH, torch.device("cuda:0"))
    annotations = dict()
    for f in tqdm(os.listdir(os.path.join("lizard", "images", split))):
        detections = inference([read_png_into_numpy(os.path.join("lizard", "images", split, f))])
        detections = detections[0].detach().cpu().numpy()
        detections[:, :-1] /= image_size
        if len(detections.shape) == 1:
            detections = np.expand_dims(detections, axis=0)
        elif len(detections.shape) != 2:
            raise ValueError
        detections[:, 2] -= detections[:, 0]
        detections[:, 3] -= detections[:, 1]        
        annotations[os.path.join("lizard", "images", split, f)] = []
        for row in detections:
            annotations[os.path.join("lizard", "images", split, f)].append(
                {"bbox": row[:-1].tolist(), "label": "neutrophil", "score": row[-1]}
            )
    return annotations


def main():
    split = "train"
    export_dir = f"pred_{split}"
    create_coco_like_json(
        create_pred_annotations(split, 256),
        split,
        export_dir,
    )


if __name__ == "__main__":
    main()
