import constants
import fire
import model_wrappers
import numpy as np
import os
import pandas as pd
import torch
import utils
import yake

import nltk

nltk.download("punkt")
nltk.download("stopwords")


def extract_keywords(texts: list[str], kw_lens: list[int], top=256):
    """
    texts - list[str]: collection of texts to extract the keywords from
    kw_lens - list[int]: sequence of upper bounds on the length of the keywords
    top - int
    """
    # print(texts[141037]) has a nan
    all_texts = " .\n".join([str(t) for t in texts])
    stopwords_ = set(nltk.corpus.stopwords.words("english"))
    all_keywords = []
    for l in kw_lens:
        kw_extractor = yake.KeywordExtractor(
            lan="en", n=l, dedupLim=0.9, top=top, features=None
        )
        keywords = kw_extractor.extract_keywords(all_texts)
        keywords = [keyword[0] for keyword in keywords]
        all_keywords += keywords
    all_keywords = [k.replace(".", "").strip() for k in all_keywords]
    all_keywords = list(set(all_keywords) - stopwords_)

    return all_keywords


def get_captions_keywords_per_cls(dataset, top=256, only_spurious=False, remake=False):
    ending = "_only_spurious" if only_spurious else ""
    kw_cache_dir = os.path.join(
        constants.CACHE_PATH,
        "keywords",
        dataset,
    )
    os.makedirs(kw_cache_dir, exist_ok=True)

    kw_cache_file = os.path.join(kw_cache_dir, f"keywords{ending}.npy")
    if not remake and os.path.isfile(kw_cache_file):
        kws_dict = np.load(kw_cache_file, allow_pickle=True).item()
        class_keywords = [
            kws_dict[i] for i in range(len(constants.DATASET_CLASSES[dataset]))
        ]
        return class_keywords

    captions_path = os.path.join(constants.CACHE_PATH, "captions", dataset, "train.txt")
    if not os.path.exists(captions_path):
        raise FileNotFoundError(captions_path)
    f = open(captions_path, "r")
    captions = f.read().split("\n")
    f.close()

    metadata_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR[dataset],
        constants.METADATA_NAME[dataset],
    )
    mtd = pd.read_csv(metadata_path)
    train_labels = mtd.y[mtd.split == constants.DATASET_SPLITS["train"]].to_list()
    train_envs = mtd.a[mtd.split == constants.DATASET_SPLITS["train"]].to_list()
    del mtd
    class_keywords = []
    if only_spurious:
        for cls in range(len(constants.DATASET_CLASSES[dataset])):
            class_captions = [
                c
                for c, l, env in zip(captions, train_labels, train_envs)
                if l == cls and env == constants.DATASET_CLASS_BIAS[dataset][cls]
            ]
            class_keywords.append(
                extract_keywords(class_captions, kw_lens=[3, 5], top=top)
            )
    else:
        for cls in range(len(constants.DATASET_CLASSES[dataset])):
            class_captions = [c for c, l in zip(captions, train_labels) if l == cls]
            class_keywords.append(
                extract_keywords(class_captions, kw_lens=[3, 5], top=top)
            )

    kws_dict = {i: kws for i, kws in enumerate(class_keywords)}
    np.save(kw_cache_file, kws_dict)
    return class_keywords


def get_civilcomments_keywords_per_cls(top=256, only_spurious=False, remake=False):
    dataset = "CivilComments"
    ending = "_only_spurious" if only_spurious else ""

    kw_cache_dir = os.path.join(
        constants.CACHE_PATH,
        "keywords",
        dataset,
    )
    os.makedirs(kw_cache_dir, exist_ok=True)
    kw_cache_file = os.path.join(kw_cache_dir, f"keywords{ending}.npy")
    if not remake and os.path.isfile(kw_cache_file):
        kws_dict = np.load(kw_cache_file, allow_pickle=True).item()
        class_keywords = [
            kws_dict[i] for i in range(len(constants.DATASET_CLASSES[dataset]))
        ]
        return class_keywords

    csv_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR["CivilComments"],
        "civilcomments_coarse.csv",
    )
    metadata_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR["CivilComments"],
        constants.METADATA_NAME["CivilComments"],
    )

    df = pd.read_csv(csv_path)
    mtd = pd.read_csv(metadata_path)
    texts = df.comment_text[mtd.split == constants.DATASET_SPLITS["train"]].to_numpy()
    train_labels = mtd.y[mtd.split == constants.DATASET_SPLITS["train"]].to_numpy()
    train_envs = mtd.a[mtd.split == constants.DATASET_SPLITS["train"]].to_numpy()
    del mtd
    class_keywords = []
    if only_spurious:
        for cls in range(len(constants.DATASET_CLASSES[dataset])):
            class_texts = texts[
                (train_labels == cls)
                & (train_envs == constants.DATASET_CLASS_BIAS[dataset][cls])
            ]
            class_keywords.append(
                extract_keywords(class_texts, kw_lens=[3, 5], top=top)
            )
    else:
        for cls in range(len(constants.DATASET_CLASSES[dataset])):
            class_texts = texts[train_labels == cls]
            class_keywords.append(
                extract_keywords(class_texts, kw_lens=[3, 5], top=top)
            )

    kws_dict = {i: kws for i, kws in enumerate(class_keywords)}
    np.save(kw_cache_file, kws_dict)
    return class_keywords


def cache_clip_keywords_and_embeddings_per_cls(dataset: str, only_spurious=False):
    """ """
    class_keywords = get_captions_keywords_per_cls(
        dataset, only_spurious=only_spurious, remake=False
    )
    all_keywords = list(set(sum(class_keywords, start=[])))
    # remove class_instances
    ending = "_only_spurious" if only_spurious else ""
    processed_keywords_path = os.path.join(
        constants.CACHE_PATH, "keywords", dataset, f"filtered_keywords{ending}.pt"
    )
    print(processed_keywords_path)

    if os.path.isfile(processed_keywords_path):
        saved_keywords = torch.load(processed_keywords_path)
        clean_keywords = saved_keywords["clean"]
    else:
        llm_keywords, llm_raw_keywords = utils.remove_class_instances_llm(
            all_keywords,
            constants.DATASET_CLASSES_EXPLICIT[dataset],
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            weights_dtype=torch.bfloat16,
            device_map="cuda",
            # to_remove=dataset + so,
        )
        _, compound_terms, class_attributes, _ = utils.remove_class_instances(
            llm_keywords,
            constants.DATASET_CLASSES[dataset],
            constants.DATASET_CLASSES_WN[dataset],
        )
        clean_keywords = list(set(compound_terms + class_attributes))

        kwd = {
            "llm_raw": llm_raw_keywords,
            "llm": llm_keywords,
            "clean": clean_keywords,
        }
        torch.save(kwd, processed_keywords_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_encoder = model_wrappers.CLIPTextEncoderWrapper()

    keywords_embeddings = clip_encoder.encode_texts_batched(
        clean_keywords, device=device
    )

    d = {
        "keywords": clean_keywords,
        "keywords_embeddings": keywords_embeddings,
    }

    output_path = os.path.join(constants.CACHE_PATH, "biases", dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.save(
        d, os.path.join(output_path, f"filtered_keywords_and_embeddings{ending}.pt")
    )


def cache_civilcomments_keywords_and_embeddings_per_cls(only_spurious=False):
    """ """
    dataset = "CivilComments"
    class_keywords = get_civilcomments_keywords_per_cls(
        only_spurious=only_spurious, remake=False
    )
    all_keywords = list(set(sum(class_keywords, start=[])))
    ending = "_only_spurious" if only_spurious else ""
    processed_keywords_path = os.path.join(
        constants.CACHE_PATH, "keywords", dataset, f"filtered_keywords{ending}.pt"
    )

    if os.path.isfile(processed_keywords_path):
        saved_keywords = torch.load(processed_keywords_path)
        clean_keywords = saved_keywords["clean"]
    else:
        llm_keywords, llm_raw_output = utils.remove_class_instances_llm(
            all_keywords,
            constants.DATASET_CLASSES_EXPLICIT[dataset],
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            weights_dtype=torch.bfloat16,
            device_map="cuda",
            # to_remove=dataset + so
        )
        _, compound_terms, class_attributes, _ = utils.remove_class_instances(
            llm_keywords,
            constants.DATASET_CLASSES[dataset],
            constants.DATASET_CLASSES_WN[dataset],
        )
        clean_keywords = list(set(compound_terms + class_attributes))

        kwd = {
            "llm_raw": llm_raw_output,
            "llm": llm_keywords,
            "clean": clean_keywords,
        }
        torch.save(kwd, processed_keywords_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model_wrappers.SentenceEncoderWrapper()

    keywords_embeddings = encoder.encode_texts_batched(clean_keywords, device=device)

    d = {
        "keywords": clean_keywords,
        "keywords_embeddings": keywords_embeddings,
    }
    output_path = os.path.join(constants.CACHE_PATH, "biases", dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.save(
        d, os.path.join(output_path, f"filtered_keywords_and_embeddings{ending}.pt")
    )


def cache_keywords_and_embeddings(dataset: str, only_spurious=False):
    if dataset == "CivilComments":
        cache_civilcomments_keywords_and_embeddings_per_cls(only_spurious)
    else:
        cache_clip_keywords_and_embeddings_per_cls(dataset, only_spurious)


if __name__ == "__main__":
    fire.Fire(cache_keywords_and_embeddings)
