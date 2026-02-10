from torch.utils.data import DataLoader
from tqdm import tqdm
import constants
import fire
import model_wrappers
import numpy as np
import os
import pandas as pd
import templates
import torch
import torch.nn as nn
import utils


def zero_shot_with_bias_enhanced_prompts(
    dataset: str = "Waterbirds",
    device: str = "cuda",
    alt_source: None | str = None,  # 'b2t', 'lg', 'splice'
    top_x: None | int = None,  # 30
    top_p: None | float = None,  # 0.2
    SEED: None | int = None,
):
    if SEED is None:
        SEED = constants.SEED

    metadata_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR[dataset],
        constants.METADATA_NAME[dataset],
    )
    mtd = pd.read_csv(metadata_path)
    labels = {
        split: mtd.y[mtd.split == constants.DATASET_SPLITS[split]]
        for split in constants.SPLITS
    }
    envs = {
        split: mtd.a[mtd.split == constants.DATASET_SPLITS[split]]
        for split in constants.SPLITS
    }

    embeddings_dir = os.path.join(
        constants.CACHE_PATH,
        "model_outputs",
        dataset,
    )
    embeddings = {}
    for split in constants.SPLITS:
        emb_path = os.path.join(embeddings_dir, f"{split}.npy")
        embeddings[split] = utils.normalize_embeddings(np.load(emb_path))

    biases_set = set()
    if alt_source is not None:
        for cls in constants.DATASET_CLASSES[dataset]:
            class_biases_path = os.path.join(
                constants.CACHE_PATH, "biases", dataset, f"{cls}_biases_{alt_source}.pt"
            )
            class_biases = torch.load(class_biases_path)
            biases_set.update(class_biases["biases"])
    else:
        if top_x is not None:
            biases = utils.get_top_biases(
                dataset,
                ending="",
                top=top_x,
            )
            biases_set.update(biases)
        elif top_p is not None:
            biases = utils.get_top_biases(
                dataset,
                ending="",
                top_percent=top_p,
            )
            biases_set.update(biases)
        else:
            for cls in constants.DATASET_CLASSES[dataset]:
                class_biases_path = os.path.join(
                    constants.CACHE_PATH, "biases", dataset, f"{cls}_biases.pt"
                )
                class_biases = torch.load(class_biases_path)
                biases_set.update(class_biases["biases"])
    print(biases_set)
    prompts = []
    for cls in constants.DATASET_CLASSES[dataset]:
        for bias in biases_set:
            for tmp in templates.TEMPLATES[dataset]:
                prompts.append(tmp.format(bias=bias, cls=cls))

    cls_prompts = len(templates.TEMPLATES[dataset]) * len(biases_set)

    if dataset == "CivilComments":
        encoder = model_wrappers.SentenceEncoderWrapper()
    else:
        encoder = model_wrappers.CLIPTextEncoderWrapper()

    prompts_embeddings = encoder.encode_texts_batched(prompts, device=device, bs=256)
    prompts_embeddings /= prompts_embeddings.norm(p=2, dim=1, keepdim=True)
    prompts_embeddings = prompts_embeddings.numpy()

    if alt_source is not None:
        print(alt_source, "biases")
    else:
        print("WASP biases")
        if top_x is not None:
            print("Top", top_x, "biases")
        elif top_p is not None:
            print("Top ", round(top_p * 100, 2), "% biases", sep="")

    results = {}
    # for split in constants.SPLITS:
    for split in ["test"]:
        classes = np.unique(labels[split])
        uenvs = np.unique(envs[split])

        split_results = {}
        dots = prompts_embeddings @ embeddings[split].T
        predictions = np.argmax(dots, axis=0) // cls_prompts

        split_results["acc"] = np.mean(predictions == labels[split])
        split_results["group_wise_acc"] = {}
        for e in uenvs:
            mask_env = envs[split] == e
            for cls in classes:
                mask_cls = labels[split] == cls
                split_results["group_wise_acc"][f"{cls}_{e}"] = np.mean(
                    predictions[mask_env & mask_cls] == cls
                )

        split_results["worst_group_acc"] = np.min(
            list(split_results["group_wise_acc"].values())
        )
        print(split)
        print("Average Accuracy:", round(100 * split_results["acc"], 2))
        print("Worst Group Accuracy:", round(100 * split_results["worst_group_acc"], 2))

        results[split] = split_results


if __name__ == "__main__":
    fire.Fire(zero_shot_with_bias_enhanced_prompts)
