from sklearn.metrics import balanced_accuracy_score
import constants
import fire
import model_wrappers
import numpy as np
import os
import pandas as pd
import templates
import utils
import templates


def zero_shot(dataset: str, device="cuda"):
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

    classes = np.unique(labels["train"])

    prompts = []
    for cls in constants.DATASET_CLASSES[dataset]:
        prompts.append(templates.BASE_TEMPLATE[dataset].format(cls=cls))

    if dataset == "CivilComments":
        encoder = model_wrappers.SentenceEncoderWrapper()
    else:
        encoder = model_wrappers.CLIPTextEncoderWrapper()

    prompts_embeddings = encoder.encode_texts_batched(prompts, device=device, bs=128)
    prompts_embeddings /= prompts_embeddings.norm(p=2, dim=1, keepdim=True)
    prompts_embeddings = prompts_embeddings.numpy()

    results = {}
    # for split in constants.SPLITS:
    for split in ["test"]:
        classes = np.unique(labels[split])
        uenvs = np.unique(envs[split])

        split_results = {}
        dots = prompts_embeddings @ embeddings[split].T
        predictions = np.argmax(dots, axis=0)

        split_results["acc"] = np.mean(predictions == labels[split])
        split_results["balanced_acc"] = balanced_accuracy_score(
            labels[split], predictions
        )
        split_results["group_wise_acc"] = {}
        for e in uenvs:
            mask_env = envs[split] == e
            for cls in classes:
                mask_cls = labels[split] == cls
                split_results["group_wise_acc"][f"{cls}_{e}"] = np.mean(
                    predictions[mask_env & mask_cls]
                    == labels[split][mask_env & mask_cls]
                )

        split_results["worst_group_acc"] = np.min(
            list(split_results["group_wise_acc"].values())
        )

        print(split)
        print("Average Accuracy:", round(100 * split_results["acc"], 2))
        print("Worst Group Accuracy:", round(100 * split_results["worst_group_acc"], 2))

        results[split] = split_results


if __name__ == "__main__":
    fire.Fire(zero_shot)
