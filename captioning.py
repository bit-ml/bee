from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import constants
import datasets_local
import logging
import os
import sys
import torch


def caption(dataset):
    """
    Caption dataset using microsoft git
    """
    logging.info("\n ### Captioning dataset")

    # setting output path
    output_folder = os.path.join(constants.CACHE_PATH, "captions", dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # checking for existing captions
    splits = ["train"]
    n_found = 0
    for split in splits:
        output_path = os.path.join(output_folder, f"{split}.txt")
        if os.path.exists(output_path):
            logging.info(f"Found {split} captions at {output_path}")
            n_found += 1
    if n_found == 1:
        return

    # loading model
    checkpoint = "microsoft/git-large-coco"
    logging.info(f"Loading captioning model: {checkpoint}")
    processor = AutoProcessor.from_pretrained(checkpoint)
    preprocess = lambda x: processor(images=x, return_tensors="pt").pixel_values[0]
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    # model = torch.compile(model)

    # loading dataset
    dataset = {
        split: datasets_local.ImageDataset(dataset, split=split, transform=preprocess)
        for split in splits
    }

    # captioning data
    # for split in splits:
    for split in ["train"]:
        output_path = os.path.join(output_folder, f"{split}.txt")
        if os.path.exists(output_path):
            continue
        logging.info(f"Captioning: {split}")
        dataloader = torch.utils.data.DataLoader(
            dataset[split], batch_size=32, shuffle=False
        )
        with open(output_path, "w") as f:
            start = False
            for _, images, _, _ in tqdm(dataloader):
                if start:
                    f.write("\n")
                start = True
                images = images.to(device)
                with torch.no_grad():
                    output = model.generate(pixel_values=images, max_length=50)
                output = processor.batch_decode(output, skip_special_tokens=True)
                f.write("\n".join(output))
