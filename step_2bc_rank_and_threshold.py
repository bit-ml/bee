import constants
import fire
import model_wrappers
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import utils


def rank_keywords(
    dataset,
    device="cuda",
    only_spurious=False,
):

    # load trained weights
    ending = "_only_spurious" if only_spurious else ""
    classifier_path = os.path.join(
        constants.CACHE_PATH, "classifiers", dataset, f"erm_classifier{ending}.pt"
    )
    state_dict = torch.load(classifier_path, weights_only=True)
    device = torch.device(device)
    weights = state_dict["weight"]
    weights = weights.to(device)

    # load keywords and their embeddings
    kw_path = os.path.join(
        constants.CACHE_PATH,
        "biases",
        dataset,
        f"filtered_keywords_and_embeddings{ending}.pt",
    )
    kw_dict = torch.load(kw_path)
    keywords_embeddings = kw_dict["keywords_embeddings"]
    keywords_embeddings_normed = keywords_embeddings / keywords_embeddings.norm(
        p=2, dim=1, keepdim=True
    )
    keywords_embeddings_normed = keywords_embeddings_normed.to(device)
    keywords = np.array(kw_dict["keywords"])

    output_dir = os.path.join(
        constants.CACHE_PATH,
        "biases",
        dataset,
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        similarities = weights @ keywords_embeddings_normed.T
    similarities = similarities.cpu().numpy()

    max_similarity = np.max(similarities, axis=0)  # max across classes
    min_similarity = np.min(similarities, axis=0)  # min across classes
    gap = max_similarity - min_similarity
    gap_args = np.argsort(gap)[::-1]
    sorted_biases = keywords[gap_args]
    sorted_similarities = similarities[:, gap_args]

    df = {"bias": sorted_biases, "score": gap[gap_args]}
    for idx, cls in enumerate(constants.DATASET_CLASSES[dataset]):
        df[cls] = sorted_similarities[idx]

    df = pd.DataFrame.from_dict(df)
    gap_df_path = os.path.join(output_dir, f"max_gap_ranking{ending}.csv")
    df.to_csv(gap_df_path, index=False)

    # class specific biases
    for idx, cls in enumerate(constants.DATASET_CLASSES[dataset]):
        class_similarity = similarities[idx]

        similarity_diff = class_similarity - min_similarity
        class_args = np.argsort(similarity_diff)[::-1].copy()
        sorted_diffs = similarity_diff[class_args]
        sorted_biases = keywords[class_args]
        sorted_similarities = similarities[:, class_args]

        df = {
            "bias": sorted_biases,
            "score": sorted_diffs,
        }
        for idx_, cls_ in enumerate(constants.DATASET_CLASSES[dataset]):
            df[cls_] = sorted_similarities[idx_]

        df = pd.DataFrame.from_dict(df)
        class_df_path = os.path.join(output_dir, f"{cls}_ranking{ending}.csv")
        df.to_csv(class_df_path, index=False)

        scores = sorted_diffs[sorted_diffs > 0]
        threshold, arg = utils.threshold_bias_scores_convolve(scores, return_arg=True)
        print(threshold, arg)

        class_biases = sorted_biases[
            : arg + 1
        ]  # includes the concept which gives the threshold
        class_bias_embeddings = keywords_embeddings[class_args][: arg + 1]
        biases_dict = {
            "biases": class_biases,
            "bias_embeddings": class_bias_embeddings,
        }
        class_biases_path = os.path.join(output_dir, f"{cls}_biases{ending}.pt")
        torch.save(biases_dict, class_biases_path)

        # plot the scores for each classe
        fig, ax = plt.subplots()
        ax.plot(sorted_diffs)
        ax.set_ylabel("Bias score")
        ax.set_xticks([])
        # ax.axhline(y=0, color='k')
        ax.axhline(y=threshold, color="#ff5842", label="Bias Threshold")
        ax.set_title(cls)
        ax.legend()
        fig.savefig(os.path.join(output_dir, f"{cls}_bias_scores{ending}.png"))
        plt.close(fig)


if __name__ == "__main__":
    fire.Fire(rank_keywords)
