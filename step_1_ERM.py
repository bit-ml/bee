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
from sklearn.utils.class_weight import compute_class_weight


def train_layer(
    dataset="Waterbirds",
    device="cuda",
    BS=1024,
    only_spurious=False,
):
    np.random.seed(constants.SEED)
    torch.random.manual_seed(constants.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ending = ""
    if only_spurious:
        ending += "_only_spurious"
        datasets = utils.load_dataset_splits_only_spurious(dataset, env_aware=False)
    else:
        datasets = utils.load_dataset_splits(dataset, env_aware=False)

    device = torch.device(device)

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

    dataloaders = {
        split: DataLoader(
            datasets[split], batch_size=BS, shuffle=split == "train", drop_last=False
        )
        for split in constants.SPLITS
    }

    output_dir = os.path.join(constants.CACHE_PATH, "classifiers", dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    classifier_path = os.path.join(output_dir, f"erm_classifier{ending}.pt")

    classifier = model_wrappers.TemperatureScaledLinearLayer(
        input_dim=input_dim, output_dim=n_classes, init_temp=constants.TEMP_INIT
    )
    classifier.temperature.requires_grad_(False)
    with torch.no_grad():
        classifier.weight.data.copy_(layer_init_weights)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = None

    best_val_acc = 0
    patience = 5
    ne = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    pbar = tqdm(total=float("inf"), desc="ERM Training")
    while True:
        train_loss, train_acc = utils.train(
            classifier, device, dataloaders["train"], loss_fn, optimizer, scheduler
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
    torch.save(details, os.path.join(output_dir, f"erm_metrics{ending}.pt"))


if __name__ == "__main__":
    fire.Fire(train_layer)
