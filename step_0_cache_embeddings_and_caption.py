import captioning
import constants
import datasets_local
import fire
import model_wrappers
import numpy as np
import open_clip
import os
import pandas as pd
import torch
import utils


def cache_embeddings_and_caption_dataset(
    dataset="Waterbirds",
    clip: str = "openai/ViT-L-14",
    device: str = "cuda",
):
    """
    Captions dataset to create a vocabulary.
    Loads the initial and finetuned classification weights.
    Computes biases based on the weights and the word embeddings.
    """
    output_dir = os.path.join(constants.CACHE_PATH, "model_outputs", dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device(device)

    clip_version, clip_architecture = clip.split("/")
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_architecture, pretrained=clip_version
    )
    model.eval()
    model.to(device)
    datasets = {
        split: datasets_local.ImageDataset(dataset, split=split, transform=preprocess)
        for split in constants.SPLITS
    }

    for split in constants.SPLITS:
        utils.cache_and_get_model_outputs(
            model=model.encode_image, dataset=datasets[split], device=device
        )

    # captioning dataset
    captioning.caption(dataset)

    # cache class name embeddings
    class_names = constants.DATASET_CLASSES[dataset]
    class_embeddings = []
    tokenizer = open_clip.get_tokenizer(clip_architecture)
    with torch.no_grad():
        for idx in range(0, len(class_names), 128):
            input = tokenizer(class_names[idx : min(idx + 128, len(class_names))]).to(
                device
            )
            batch_embeddings = model.encode_text(input).cpu().numpy()
            class_embeddings.append(batch_embeddings)

    class_embeddings = np.concatenate(class_embeddings, axis=0)
    class_embeddings_path = os.path.join(output_dir, "class_embeddings.npy")
    np.save(class_embeddings_path, class_embeddings)


def cache_embeddings_civilcomments(device: str = "cuda"):
    dataset = "CivilComments"
    comments_path = os.path.join(
        constants.DATA_PATH, constants.DATASET_DIR[dataset], "civilcomments_coarse.csv"
    )
    df = pd.read_csv(comments_path)
    texts = df.comment_text.to_numpy()

    metadata_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR[dataset],
        constants.METADATA_NAME[dataset],
    )

    mtd = pd.read_csv(metadata_path)
    split_labels = mtd.split.to_numpy()

    device = torch.device(device)
    model = model_wrappers.SentenceEncoderWrapper()
    model.eval()
    model.to(device)

    output_dir = os.path.join(
        constants.CACHE_PATH,
        "model_outputs",
        dataset,
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # cache data embeddings
    for split in constants.SPLITS:
        output_path = os.path.join(output_dir, f"{split}.npy")
        if not os.path.isfile(output_path):
            split_comments = texts[split_labels == constants.DATASET_SPLITS[split]]
            split_embeddings = model.encode_texts_batched(
                split_comments, device, bs=128
            ).numpy()

            np.save(output_path, split_embeddings)

    # cache class name embeddings
    class_names = constants.DATASET_CLASSES[dataset]
    class_embeddings = model.encode_texts_batched(class_names, device, bs=128).numpy()
    class_embeddings_path = os.path.join(output_dir, "class_embeddings.npy")
    np.save(class_embeddings_path, class_embeddings)


def cache_embeddings_(
    dataset="Waterbirds",
    clip: str = "openai/ViT-L-14",
    device: str = "cuda",
):
    if dataset == "CivilComments":
        cache_embeddings_civilcomments(device)
    else:
        cache_embeddings_and_caption_dataset(dataset, clip, device)


if __name__ == "__main__":
    fire.Fire(cache_embeddings_)
