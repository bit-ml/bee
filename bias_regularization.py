import os

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
import model_wrappers
import utils


def train_layer_with_bias_regularization(
    dataset="Waterbirds",
    device="cuda",
    alpha=0.1,
    only_spurious=False,
    BS=1024,
    SEED=None,
    random_biases=False,
):
    if SEED is None:
        SEED = constants.SEED
    print(SEED)
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ending = ""
    if only_spurious:
        ending += "_only_spurious"
        datasets = utils.load_dataset_splits_only_spurious(dataset, env_aware=False)
    else:
        datasets = utils.load_dataset_splits(dataset, env_aware=False)

    datasets_eval = utils.load_dataset_splits(dataset, env_aware=True)

    device = torch.device(device)

    dataloaders = {
        split: DataLoader(
            datasets[split], batch_size=BS, shuffle=split == "train", drop_last=False
        )
        for split in constants.SPLITS
    }
    dataloaders_eval = {
        split: DataLoader(
            datasets_eval[split],
            batch_size=BS,
            shuffle=split == "train",
            drop_last=False,
        )
        for split in constants.SPLITS
    }

    train_labels = datasets["train"].labels
    classes = np.unique(train_labels)
    n_classes = classes.shape[0]
    input_dim = datasets["train"].embeddings.shape[-1]

    ## compute class_weights
    class_weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = torch.Tensor(class_weights).to(device)

    ## load class_embeddings for layer init
    class_embeddings_path = os.path.join(
        constants.CACHE_PATH, "model_outputs", dataset, "class_embeddings.npy"
    )
    class_embeddings = np.load(class_embeddings_path)

    layer_init_weights = torch.tensor(class_embeddings)
    layer_init_weights /= layer_init_weights.norm(p=2, dim=1, keepdim=True)

    biases_set = set()
    biases = []
    bias_embeddings = []
    if random_biases:
        n_biases_per_class = []
        for cls in constants.DATASET_CLASSES[dataset]:
            class_biases_path = os.path.join(
                constants.CACHE_PATH, "biases", dataset, f"{cls}_biases{ending}.pt"
            )
            class_biases = torch.load(class_biases_path)
            n_biases_per_class.append(len(class_biases["biases"]))
        biases = utils.get_random_biases(
            dataset,
            ending,
            num_biases=n_biases_per_class,
            SEED=SEED,
        )
        biases_set.update(biases)
        kw_path = os.path.join(
            constants.CACHE_PATH,
            "biases",
            dataset,
            f"filtered_keywords_and_embeddings{ending}.pt",
        )
        keywords = torch.load(kw_path)
        for kw, kw_embedding in zip(
            keywords["keywords"], keywords["keywords_embeddings"]
        ):
            if kw in biases_set:
                bias_embeddings.append(kw_embedding)
    else:
        for cls in constants.DATASET_CLASSES[dataset]:
            class_biases_path = os.path.join(
                constants.CACHE_PATH, "biases", dataset, f"{cls}_biases{ending}.pt"
            )
            class_biases = torch.load(class_biases_path)
            for bias, bias_embedding in zip(
                class_biases["biases"], class_biases["bias_embeddings"]
            ):
                if bias not in biases_set:
                    biases_set.add(bias)
                    biases.append(bias)
                    bias_embeddings.append(bias_embedding)

    biases = np.array(biases)
    bias_embeddings = torch.stack(bias_embeddings, dim=0)
    print(bias_embeddings.shape)
    bias_embeddings /= bias_embeddings.norm(p=2, dim=1, keepdim=True)

    classifier = model_wrappers.TemperatureScaledLinearLayer(
        input_dim=input_dim, output_dim=n_classes, init_temp=constants.TEMP_INIT
    )
    classifier.temperature.requires_grad_(False)
    with torch.no_grad():
        classifier.weight.data.copy_(layer_init_weights)

    output_dir = os.path.join(constants.CACHE_PATH, "classifiers", dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    classifier_path = os.path.join(
        output_dir, f"erm_bias_regularization_classifier{ending}.pt"
    )
    debiasing_loss = lambda *args, **kwargs: alpha * utils.debiasing_loss_l2(
        classifier, bias_embeddings, device=device
    )

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = None

    best_val_acc = 0
    patience = 5
    ne = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    pbar = tqdm(total=float("inf"), desc="ERM Training + debiasing loss")
    while True:
        train_loss, train_acc = utils.train(
            classifier,
            device,
            dataloaders["train"],
            loss_fn,
            optimizer,
            scheduler,
            extra_loss=debiasing_loss,
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = utils.validate(
            classifier, device, dataloaders["val"], loss_fn
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        pbar.update(1)
        if val_acc > best_val_acc:
            ne = 0
            best_val_acc = val_acc
            torch.save(classifier.cpu().state_dict(), classifier_path)
        else:
            ne += 1
            if ne == patience:
                break
    pbar.close()

    details = {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
    }
    torch.save(
        details, os.path.join(output_dir, f"erm_bias_regularization_metrics{ending}.pt")
    )

    classifier.load_state_dict(torch.load(classifier_path, weights_only=True))

    loss, wga, balanced_acc, avg_acc = utils.validate_wga(
        classifier,
        device,
        dataloaders_eval["test"],
        nn.functional.cross_entropy,
        return_balanced_acc=True,
        return_avg_acc=True,
        use_tqdm=False,
    )

    print("Test set results")
    print("Average Accuracy:", round(avg_acc * 100, 2))
    print("Class Balanced Accuracy:", round(balanced_acc * 100, 2))
    print("Worst Group Accuracy:", round(wga * 100, 2))


if __name__ == "__main__":
    fire.Fire(train_layer_with_bias_regularization)
