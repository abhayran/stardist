import numpy as np
import os
from PIL import Image
from models.experimental import attempt_load
from operator import attrgetter
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from typing import List


ORTHOPEDIA_DIR = "/mnt/dss/dssfs04/pn25ke/pn25ke-dss-0001/orthopedia"
MODEL_PATH = "/mnt/dss/dsshome1/09/ge59gew2/orthopedia/yolov7_neutrophil.pt"
DEVICE_ID = 0
BATCH_SIZE = 100


class Embedder:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = attempt_load(MODEL_PATH).eval().to(self.device)
        
        self.features = {}
        attrgetter("model.105.m.0")(self.model).register_forward_hook(self.get_features('out_256'))
        attrgetter("model.105.m.1")(self.model).register_forward_hook(self.get_features('out_512'))
        attrgetter("model.105.m.2")(self.model).register_forward_hook(self.get_features('out_1024'))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def get_features(self, name: str):
        def hook(model, input_, output_):
            self.features[name] = output_.detach()
        return hook
        
    @torch.no_grad()
    def __call__(self, image_paths: List[str]):
        input_tensor = Variable(
            torch.stack([
                self.transform(Image.open(path))
                for path in image_paths
            ]).to(self.device)
        )
        self.model(input_tensor)
        return [
            self.features['out_256'],
            self.features['out_512'],
            self.features['out_1024'],
        ]


def map_path(path: str) -> str:
    return path.replace("tiles", "embedding").replace(ORTHOPEDIA_DIR, "/mnt/dss/dsshome1/09/ge59gew2")


def make_embedding_dir(embedding_path: str) -> None:
    path_to_make = map_path(embedding_path)
    if not os.path.exists(path_to_make):
        os.mkdir(path_to_make)


def main():
    device = torch.device("cpu") if DEVICE_ID < 0 else torch.device(f"cuda:{DEVICE_ID}") 
    embedder = Embedder(device)

    for key_folder in ["infect", "noinfect"]:

        key_path = os.path.join(ORTHOPEDIA_DIR, f"{key_folder}_tiles")
        make_embedding_dir(key_path)
        
        for fn_folder in os.listdir(key_path):

            fn_path = os.path.join(key_path, fn_folder)
            make_embedding_dir(fn_path)

            for img_folder in os.listdir(fn_path):
                img_path = os.path.join(fn_path, img_folder)                
                img_files = os.listdir(img_path)
                
                if len(img_files) != 1000:
                    print("Skipping image ID:", img_folder)
                    continue
                    
                print("Running inference on image ID:", img_folder)
                num_iters = len(img_files) // BATCH_SIZE + (len(img_files) % BATCH_SIZE > 0)
                
                embeddings_256 = list()
                embeddings_512 = list()
                embeddings_1024 = list()
                
                for i in range(num_iters):
                    embeddings = embedder([
                        os.path.join(img_path, item)
                        for item in img_files[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                    ])
                    embeddings_256.append(embeddings[0])
                    embeddings_512.append(embeddings[1])
                    embeddings_1024.append(embeddings[2])
                
                np.savez_compressed(
                    map_path(os.path.join(fn_path, img_folder)),
                    out_256=torch.cat(embeddings_256).cpu().numpy(),
                    out_512=torch.cat(embeddings_512).cpu().numpy(),
                    out_1024=torch.cat(embeddings_1024).cpu().numpy(),
                )


if __name__ == "__main__":
    main()
