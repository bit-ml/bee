from PIL import Image
from torch.utils.data import Dataset
import constants
import numpy as np
import os
import pandas as pd

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, dataset: str, split: str, transform=None):
        self.name = dataset
        self.split = split
        self.transform = transform

        self.n_classes = len(constants.DATASET_CLASSES[dataset])

        self.root_dir = os.path.join(
            constants.DATA_PATH,
            constants.DATASET_DIR[dataset],
            constants.DATASET_IMAGE_DIR[dataset],
        )

        self.metadata = pd.read_csv(
            os.path.join(
                constants.DATA_PATH,
                constants.DATASET_DIR[dataset],
                constants.METADATA_NAME[dataset],
            )
        )
        metadata_split = self.metadata.split.to_numpy()
        self.paths = self.metadata.filename.to_numpy()[
            metadata_split == constants.DATASET_SPLITS[split]
        ]
        self.labels = self.metadata.y.to_numpy(dtype=np.int32)[
            metadata_split == constants.DATASET_SPLITS[split]
        ]
        try:
            self.envs = self.metadata.a.to_numpy(dtype=np.int32)[
                metadata_split == constants.DATASET_SPLITS[split]
            ]
        except:
            self.envs = np.zeros_like(self.labels)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.paths[index])
        image = Image.open(path).convert("RGB")
        env = self.envs[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return index, image, env, label

    def __len__(self):
        return len(self.paths)


class EmbDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class EnvAwareEmbDataset(Dataset):
    def __init__(
        self,
        embeddings,
        labels,
        envs,
        only_spurious=False,
        return_env=False,
        bias_atts=None,
    ):
        self.only_spurious = only_spurious
        self.return_env = True

        if only_spurious:
            self.embeddings = []
            self.labels = []
            self.envs = []
            for cls, bias_att in enumerate(bias_atts):
                mask = (labels == cls) & (envs == bias_att)
                self.embeddings.append(embeddings[mask])
                self.labels.append(labels[mask])
                self.envs.append(envs[mask])

            self.embeddings = np.concatenate(self.embeddings)
            self.labels = np.concatenate(self.labels)
            self.envs = np.concatenate(self.envs)
            self.envs *= 0  # hack to make gdro work out of the box; q will have shape [n_classes]
            u_envs = np.unique(self.envs)
            mapping = {env: idx for idx, env in enumerate(u_envs)}
            self.envs = [mapping[env] for env in self.envs]
            self.return_env = return_env
        else:
            self.embeddings = embeddings
            self.labels = labels
            u_envs = np.unique(envs)
            mapping = {env: idx for idx, env in enumerate(u_envs)}
            self.envs = [mapping[env] for env in envs]
        self.envs = np.array(self.envs, dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.return_env:
            return self.embeddings[idx], self.labels[idx], self.envs[idx]

        return self.embeddings[idx], self.labels[idx]
