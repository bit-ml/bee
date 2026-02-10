from torch.utils.data import DataLoader
from tqdm import tqdm
import constants
import fire
import model_wrappers
import numpy as np
import os
import torch
import torch.nn as nn
import utils


def train_layer_gdro(
    dataset="Waterbirds",
    device="cuda",
    only_spurious=False,
    BS=1024,
    SEED=None,
):
    if SEED is None:
        SEED = constants.SEED
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ending = ""

    ## load dataset
    if only_spurious:
        ending += "_only_spurious"
        datasets = utils.load_dataset_splits_only_spurious(dataset, env_aware=True)
    else:
        datasets = utils.load_dataset_splits(dataset, env_aware=True)

    dataloaders = {
        split: DataLoader(
            datasets[split], batch_size=BS, shuffle=split == "train", drop_last=False
        )
        for split in constants.SPLITS
    }

    device = torch.device(device)

    train_labels = datasets["train"].labels
    classes = np.unique(train_labels)
    n_classes = classes.shape[0]
    input_dim = datasets["train"].embeddings.shape[-1]
    n_envs = np.unique(datasets["train"].envs).shape[0]
    q = torch.ones(n_classes * n_envs, dtype=torch.float32).to(device)

    ## load class_embeddings for layer init
    class_embeddings_path = os.path.join(
        constants.CACHE_PATH, "model_outputs", dataset, "class_embeddings.npy"
    )
    class_embeddings = np.load(class_embeddings_path)

    layer_init_weights = torch.tensor(class_embeddings)
    layer_init_weights /= layer_init_weights.norm(p=2, dim=1, keepdim=True)

    output_dir = os.path.join(constants.CACHE_PATH, "classifiers", dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    classifier_path = os.path.join(output_dir, f"gdro_classifier{ending}.pt")

    classifier = model_wrappers.TemperatureScaledLinearLayer(
        input_dim=input_dim, output_dim=n_classes, init_temp=constants.TEMP_INIT
    )
    classifier.temperature.requires_grad_(False)

    with torch.no_grad():
        classifier.weight.data.copy_(layer_init_weights)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_acc = 0
    patience = 5
    ne = 0

    train_losses = []
    train_accs = []
    train_baccs = []
    train_wgas = []
    val_losses = []
    val_accs = []
    pbar = tqdm(total=float("inf"))
    while True:
        train_loss, train_wga, train_balanced_acc, train_acc = utils.train_groupdro(
            classifier,
            device,
            dataloaders["train"],
            loss_fn,
            optimizer,
            scheduler=None,
            n_envs=n_envs,
            q=q,
            gdro_eta=1e-2,
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_baccs.append(train_balanced_acc)
        train_wgas.append(train_wga)

        val_loss, val_acc = utils.validate_wga(
            classifier, device, dataloaders["val"], nn.functional.cross_entropy
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
        "train_balanced_accs": train_baccs,
        "train_worst_accs": train_wgas,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
    }
    torch.save(details, os.path.join(output_dir, f"gdro_metrics{ending}.pt"))

    ## compute bias similarities
    classifier.load_state_dict(torch.load(classifier_path))

    loss, wga, balanced_acc, avg_acc = utils.validate_wga(
        classifier,
        device,
        dataloaders["test"],
        nn.functional.cross_entropy,
        return_balanced_acc=True,
        return_avg_acc=True,
    )
    print("Test set results")
    print("Average Accuracy:", round(avg_acc * 100, 2))
    print("Class Balanced Accuracy:", round(balanced_acc * 100, 2))
    print("Worst Group Accuracy:", round(wga * 100, 2))


if __name__ == "__main__":
    fire.Fire(train_layer_gdro)
