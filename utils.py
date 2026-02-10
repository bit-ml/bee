from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import constants
import logging
import numpy as np
import pandas as pd
import os
import time
import torch
import transformers
import yake
import datasets_local

sharing_strategy = "file_system"


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def format_seconds(s):
    """
    Transforms 128 into 00:02:08
    """
    h = int(s / 3600)
    s = s - h * 3600
    m = int(s / 60)
    s = s - m * 60
    h, m, s = map(str, [h, m, s])
    h, m, s = map(lambda x: "0" * max(0, 2 - len(x)) + x, [h, m, s])
    return f"{h}:{m}:{s}"


def normalize_embeddings(embeddings):
    return embeddings / np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))


def get_model_outputs(model, dataloader, device):
    """
    Returns a np.array featuring the model output for each data point in the dataloader
    """
    indices = []
    v_embeddings = []
    with torch.no_grad():
        logging.info("Computing outputs")
        for batch_indices, batch_imgs, _, batch_labels in tqdm(dataloader):
            batch_imgs = batch_imgs.to(device)
            batch_emb = model(batch_imgs)

            v_embeddings.append(batch_emb.cpu().numpy())
            indices.append(batch_indices.numpy())
    indices = np.concatenate(indices)
    v_embeddings = np.concatenate(v_embeddings)

    args = np.argsort(indices)
    v_embeddings = v_embeddings[
        args
    ]  # sort embeddings so that they match samples in metadata
    ## just in case some1 messed up and sent a dataloader with random sampling
    return v_embeddings


def cache_and_get_model_outputs(
    model: "torch model",
    dataset: "torch dataset",
    device: "torch device",
    batch_size=128,
):
    """
    Runs the model and saves the predictions for all data points
    If the embeddings already exist, it just reads them

    Returns
    -------
    embeddings: np.array, shape = (len(dataset), d)
    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    output_path = os.path.join(constants.CACHE_PATH, "model_outputs", dataset.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, f"{dataset.split}.npy".replace("/", "_"))
    if os.path.exists(output_path):
        logging.info(f"Loading outputs from: {output_path}")
        return np.load(output_path, allow_pickle=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=set_worker_sharing_strategy,
    )
    image_embeddings = None
    image_embeddings = get_model_outputs(
        # model.encode_image, dataloader, device
        model,
        dataloader,
        device,
    )

    np.save(output_path, image_embeddings)
    return image_embeddings


def filter_unrelated_words(reference_word, words: list[str], pos="n") -> list[str]:
    """Separates the given list of words into two lists of words that are related or not
    to the reference word by means of the hypernymy of hyponymy relationships.

    Parameters
    ------------
    reference_word: str
        The word to compare against.
    words: list[str]
        The list of words to be verified.
    pos: str
        POS tag used to reduce the number of synsets that have to be checked.
        Pass None to test every synset for every word.

    Returns:
    ------------
    unrelated_words: list[str]
        A subset of 'words' which are NOT related to 'reference_word' by means of hypernymy
        or hyponymy.
    related_words: list[str]
        A subset of 'words' which are related to 'reference_word' by means of hypernymy
        or hyponymy (i.e. the complement of 'unrelated_words' with respect to 'words').
    """

    unrelated = [True for _ in words]
    if pos is not None:
        reference_synsets = wn.synsets(reference_word, pos=pos)
        words_synsets = [wn.synsets(word, pos=pos) for word in words]
    else:
        reference_synsets = wn.synsets(reference_word)
        words_synsets = [wn.synsets(word) for word in words]

    for reference_synset in reference_synsets:
        all_hyponyms = set(reference_synset.closure(lambda x: x.hyponyms())) | {
            reference_synset
        }
        for i, word_synsets in enumerate(words_synsets):
            if unrelated[i]:
                for word_synset in word_synsets:
                    if word_synset in all_hyponyms:
                        unrelated[i] = False
                        break

        all_hypernyms = set(reference_synset.closure(lambda x: x.hypernyms()))
        for i, word_synsets in enumerate(words_synsets):
            if unrelated[i]:
                for word_synset in word_synsets:
                    if word_synset in all_hypernyms:
                        unrelated[i] = False
                        break

    unrelated_words = [word for word, unrelated_ in zip(words, unrelated) if unrelated_]
    related_words = [
        word for word, unrelated_ in zip(words, unrelated) if not unrelated_
    ]
    return unrelated_words, related_words


def remove_class_instances(
    keywords: list[str], classes: list[str], wn_classes: list[str]
):
    ## break keywords appart and filter them based on the classes
    classes_tokenized = [nltk.word_tokenize(cls) for cls in classes]
    # split classes into words -> won't filter water for waterbird; will filter blonde for blonde hair
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    keywords_tokenized = [nltk.word_tokenize(kw) for kw in keywords]
    keywords_tokenized_lem = [
        [lemmatizer.lemmatize(w.lower()) for w in kw] for kw in keywords_tokenized
    ]

    all_stop_words = set(stopwords.words("english")) | constants.STOPWORDS
    words = sum(keywords_tokenized_lem, start=[])
    words = list(set(words) - all_stop_words)

    filtered_words = []
    related_words = []
    for word in words:
        ok_ = True
        for cls_words in classes_tokenized:
            if word in cls_words:
                related_words.append(word)
                ok_ = False
                break
        if ok_:
            filtered_words.append(word)

    unrelated_words = filtered_words
    for wn_cls in wn_classes:
        unrelated_words, related_ = filter_unrelated_words(wn_cls, unrelated_words)
        related_words += related_

    # for every keyword -> if no related word in it -> keep keywords
    compound_biases = []
    bias_templates = []
    bias_attributes = []
    related_words = set(related_words)
    for kw, kw_tokens, kw_tokens_lem in zip(
        keywords, keywords_tokenized, keywords_tokenized_lem
    ):
        mask = [x not in related_words for x in kw_tokens_lem]
        non_cls_words = sum(mask)

        cw_mask = [m and (x not in all_stop_words) for x, m in zip(kw_tokens_lem, mask)]
        content_words = sum(cw_mask)

        if non_cls_words == len(
            mask
        ):  # keyword does not cotain any class-related words
            compound_biases.append(kw)

        elif content_words > 0:  # remainig string has at least 1 content word
            tokens = [x for x, m in zip(kw_tokens, mask) if m]
            bias_attributes.append(" ".join(tokens))

            if (
                non_cls_words == len(mask) - 1
            ):  # single missing word -> can make template with class names
                tokens_template = [x if m else "{}" for x, m in zip(kw_tokens, mask)]
                bias_templates.append(" ".join(tokens_template))
            # more than 2 class related words -> only extract the attributes

    return unrelated_words, compound_biases, bias_attributes, bias_templates


def post_process_llm_completion(completion, initial_seq):
    try:
        response: str = completion[0]["generated_text"][-1]["content"]
    except:
        return "", ""

    left = response.find("'")
    if left >= 0:
        right = response.find("'", left + 1)
        if right > 0:
            response = response[left + 1 : right]

    # response.replace('None', '').replace('\'', '').strip() #
    response_ = response.replace("None", "").replace("'", "").strip()

    response_split = response_.split(" ")
    initial_seq_split = initial_seq.split(" ")
    for word in response_split:
        if word not in initial_seq_split:
            return "", ""

    return response_, response


def remove_class_instances_llm(
    keywords: list[str],
    classes_description,
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    weights_dtype=torch.bfloat16,
    device_map="auto",
):

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": weights_dtype},
        device_map=device_map,
    )

    messages = [
        [
            {
                "role": "user",
                "content": constants.LLAMA_PROMPT.format(classes_description, kw),
            },
        ]
        for kw in keywords
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=10,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        temperature=1e-3,
        # batch_size=32,# right pads which is bad for decoder only due to the causal mask
    )

    llm_outputs = [
        post_process_llm_completion(output, initial)
        for output, initial in zip(outputs, keywords)
    ]
    llm_raw_output = [x[1] for x in llm_outputs]
    llm_keywords = [x[0] for x in llm_outputs]
    llm_keywords = set(llm_keywords)
    llm_keywords.discard("")
    llm_keywords = list(llm_keywords)

    # remove those that are made of only stop words; remove leading and endings of stopwords
    all_stop_words = set(stopwords.words("english")) | constants.STOPWORDS
    keywords_tokenized = [nltk.word_tokenize(kw) for kw in llm_keywords]
    processed_keywords_tokenized = []
    for kw, kw_tokenized in zip(llm_keywords, keywords_tokenized):
        mask = [word in all_stop_words for word in kw_tokenized]
        left = 0
        while left < len(mask) and mask[left]:
            left += 1
        if left < len(mask):  # non stopwords exist
            right = len(mask) - 1
            while right > left and mask[right]:
                right -= 1
            right += 1

            processed_keywords_tokenized.append(kw_tokenized[left:right])

    processed_keywords = [" ".join(pkw) for pkw in processed_keywords_tokenized]
    processed_keywords = list(set(processed_keywords))

    return processed_keywords, llm_raw_output


def extract_keywords(
    captions, deduplication_threshold=0.9, max_ngram_size=3, num_keywords=20
):
    language = "en"
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        top=num_keywords,
        features=None,
    )
    keywords = custom_kw_extractor.extract_keywords(captions)
    keywords = [keyword[0] for keyword in keywords]
    return keywords


def load_dataset_splits(dataset: str, env_aware=False):
    embeddings_dir = os.path.join(constants.CACHE_PATH, "model_outputs", dataset)
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError("Embeddings are not precomputed.")

    metadata_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR[dataset],
        constants.METADATA_NAME[dataset],
    )
    metadata = pd.read_csv(metadata_path)
    split_datasets = {}
    for split in constants.SPLITS:
        split_embeddings_path = os.path.join(embeddings_dir, f"{split}.npy")
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(
                f"Embeddings are not precomputed for {split} split."
            )
        split_embeddings = np.load(split_embeddings_path)
        split_embeddings = normalize_embeddings(split_embeddings)
        split_labels = metadata.y[
            metadata.split == constants.DATASET_SPLITS[split]
        ].to_numpy()
        if env_aware or split == "test":
            split_envs = metadata.a[
                metadata.split == constants.DATASET_SPLITS[split]
            ].to_numpy()
            split_dataset = datasets_local.EnvAwareEmbDataset(
                split_embeddings,
                split_labels,
                split_envs,
                only_spurious=False,
                return_env=True,
                bias_atts=None,
            )
        else:
            split_dataset = datasets_local.EmbDataset(split_embeddings, split_labels)
        split_datasets[split] = split_dataset

        split_envs = metadata.a[
            metadata.split == constants.DATASET_SPLITS[split]
        ].to_numpy()

    return split_datasets


def load_dataset_splits_only_spurious(dataset: str, env_aware=False):
    embeddings_dir = os.path.join(constants.CACHE_PATH, "model_outputs", dataset)
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError("Embeddings are not precomputed.")

    metadata_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR[dataset],
        constants.METADATA_NAME[dataset],
    )
    metadata = pd.read_csv(metadata_path)
    split_datasets = {}
    for split in constants.SPLITS:
        split_embeddings_path = os.path.join(embeddings_dir, f"{split}.npy")
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(
                f"Embeddings are not precomputed for {split} split."
            )
        split_embeddings = np.load(split_embeddings_path)
        split_embeddings = normalize_embeddings(split_embeddings)
        split_labels = metadata.y[
            metadata.split == constants.DATASET_SPLITS[split]
        ].to_numpy()

        split_envs = metadata.a[
            metadata.split == constants.DATASET_SPLITS[split]
        ].to_numpy()
        split_dataset = datasets_local.EnvAwareEmbDataset(
            split_embeddings,
            split_labels,
            split_envs,
            only_spurious=split != "test",
            return_env=env_aware or (split == "test"),
            bias_atts=constants.DATASET_CLASS_BIAS[dataset],
        )
        split_datasets[split] = split_dataset

        split_envs = metadata.a[
            metadata.split == constants.DATASET_SPLITS[split]
        ].to_numpy()

    return split_datasets


def worst_group_accuracy(envs, labels, predictions):
    wga_ = 1
    envs = np.array(envs)
    labels = np.array(labels)
    predictions = np.array(predictions)

    for l in np.unique(labels):
        mask1 = labels == l
        for e in np.unique(envs):
            mask2 = envs == e
            mask = mask1 & mask2
            if sum(mask) > 0:
                # print(np.mean(labels[mask] == predictions[mask]), l, e)
                wga_ = min(wga_, np.mean(labels[mask] == predictions[mask]))

    return wga_


def train(
    net,
    device,
    train_dl,
    criterion,
    optimizer,
    scheduler=None,
    extra_loss=None,
    use_tqdm=False,
):
    net.to(device)
    net.train()
    loss = 0
    num_batches = 0
    correct_preds = 0
    total_preds = 0

    labels_ = []
    predictions_ = []
    dl_iter = tqdm(train_dl) if use_tqdm else train_dl
    for batch_data, batch_labels in dl_iter:
        batch_data, batch_labels = batch_data.to(device), batch_labels.long().to(device)
        out = net(batch_data)

        batch_loss = criterion(out, batch_labels)

        if (
            extra_loss is not None
        ):  # for debiasing loss; send lamba func from upper level or partial that ignores args
            batch_loss += extra_loss(out, batch_labels)

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss += batch_loss.item()
        num_batches += 1
        total_preds += batch_data.shape[0]
        batch_predictions = torch.argmax(out, axis=-1)
        correct_preds += sum(batch_predictions.eq(batch_labels)).item()
        if scheduler is not None:
            scheduler.step()

        labels_.append(batch_labels.cpu().numpy())
        predictions_.append(batch_predictions.cpu().numpy())
        with torch.no_grad():
            net.weight.data /= net.weight.data.norm(p=2, dim=1, keepdim=True)

    labels_ = np.concatenate(labels_)
    predictions_ = np.concatenate(predictions_)
    balanced_acc = balanced_accuracy_score(labels_, predictions_)

    return loss / num_batches, balanced_acc


def validate(net, device, val_dl, criterion, use_tqdm=False):
    net.to(device)
    net.eval()
    loss = 0
    num_batches = 0
    correct_preds = 0
    total_preds = 0

    labels_ = []
    predictions_ = []

    with torch.no_grad():
        dl_iter = tqdm(val_dl) if use_tqdm else val_dl
        for batch_data, batch_labels in dl_iter:
            batch_data, batch_labels = batch_data.to(device), batch_labels.long().to(
                device
            )
            out = net(batch_data)

            batch_loss = criterion(out, batch_labels)

            loss += batch_loss.item()
            num_batches += 1
            total_preds += batch_data.shape[0]
            batch_predictions = torch.argmax(out, axis=-1)
            correct_preds += sum(batch_predictions.eq(batch_labels)).item()

            labels_.append(batch_labels.cpu().numpy())
            predictions_.append(batch_predictions.cpu().numpy())

    labels_ = np.concatenate(labels_)
    predictions_ = np.concatenate(predictions_)
    balanced_acc = balanced_accuracy_score(labels_, predictions_)

    return loss / num_batches, balanced_acc


def validate_wga(
    net,
    device,
    val_dl,
    criterion,
    return_balanced_acc=False,
    return_avg_acc=False,
    use_tqdm=False,
):
    net.to(device)
    net.eval()
    loss = 0
    num_batches = 0
    correct_preds = 0
    total_preds = 0

    labels_ = []
    predictions_ = []
    envs_ = []

    with torch.no_grad():
        dl_iter = tqdm(val_dl) if use_tqdm else val_dl
        for batch_data, batch_labels, batch_envs in dl_iter:
            batch_data, batch_labels = (
                batch_data.to(device),
                batch_labels.to(device).long(),
            )
            out = net(batch_data)

            batch_loss = criterion(out, batch_labels)

            loss += batch_loss.item()
            num_batches += 1
            total_preds += batch_data.shape[0]
            batch_predictions = torch.argmax(out, axis=-1)
            correct_preds += sum(batch_predictions.eq(batch_labels)).item()

            labels_.append(batch_labels.cpu().numpy())
            predictions_.append(batch_predictions.cpu().numpy())
            envs_.append(batch_envs.cpu().numpy())

    labels_ = np.concatenate(labels_)
    predictions_ = np.concatenate(predictions_)
    envs_ = np.concatenate(envs_)
    acc_ = worst_group_accuracy(envs_, labels_, predictions_)

    if return_balanced_acc:
        if return_avg_acc:
            return (
                loss / num_batches,
                acc_,
                balanced_accuracy_score(labels_, predictions_),
                np.mean(labels_ == predictions_),
            )
        return loss / num_batches, acc_, balanced_accuracy_score(labels_, predictions_)
    return loss / num_batches, acc_


def threshold_bias_scores(scores: np.ndarray, return_arg=False):
    if scores.shape[0] == 0:  # no biases for this cls
        return 3  # max bias score is theoretically 2;

    line = np.linspace(scores[0], scores[scores.shape[0] - 1], len(scores))
    offsets = line - scores
    arg = np.argmax(offsets)
    if return_arg:
        return scores[arg], arg

    return scores[arg]


def threshold_bias_scores_convolve(
    scores: np.ndarray, window=5, select="center", return_arg=False
):
    """
    select: str - one of center (picks the center of the window as the threshold); left, right;
            default: center; unknown selection also defaults to center
    """
    if scores.shape[0] == 0:  # no biases for this cls
        return 3 if not return_arg else 3, -1  # max bias score is theoretically 2;
    if (
        scores.shape[0] <= window
    ):  # can't convolve with valid mode; just take the first one
        if select == "left":
            return scores[0]
        elif select == "right":
            return scores[scores.shape[0] - 1]
        else:
            return scores[min(window // 2, scores.shape[0] - 1)]

    smoothed_scores = np.convolve(
        scores, np.full(window, fill_value=1 / window, dtype=np.float32), mode="valid"
    )

    score_, arg = threshold_bias_scores(smoothed_scores, return_arg=True)
    offset = 0
    if select == "left":
        offset = 0
    elif select == "right":
        offset = window - 1
    else:
        offset = window // 2
        if select != "center":
            print(
                "Threshold selection type unknown, using the default value of 'center'."
            )
    if return_arg:
        return scores[arg + offset], arg + offset

    return scores[arg + offset]


def get_top_biases(
    dataset, ending, top=30, cls_thrs=None, top_percent=None, thr_fn=None
):
    dir_path = os.path.join(constants.CACHE_PATH, "biases_v2", dataset)
    if top_percent is not None:
        assert 0 < top_percent <= 1
    elif cls_thrs is not None:
        if not isinstance(cls_thrs, list):
            cls_thrs = [cls_thrs for _ in constants.DATASET_CLASSES[dataset]]
    all_biases = []
    for i, cls in enumerate(constants.DATASET_CLASSES[dataset]):
        # if cls == 'flower': ### REMOVE LATER - single test for NICO++
        #     continue
        cls_ranking_path = os.path.join(dir_path, f"{cls}_ranking{ending}.csv")
        # print(cls_ranking_path)
        cls_ranking_df = pd.read_csv(cls_ranking_path)

        if thr_fn is not None:
            cls_scores = cls_ranking_df.score[cls_ranking_df.score > 0].to_numpy()
            cls_thr = thr_fn(cls_scores)
            cls_biases = cls_ranking_df.bias[cls_ranking_df.score >= cls_thr].to_list()

        elif top_percent is not None:
            cls_biases = cls_ranking_df.bias[cls_ranking_df.score > 0].to_list()
            cls_biases = cls_biases[: int(top_percent * len(cls_biases))]
        elif cls_thrs is not None:
            cls_biases = cls_ranking_df.bias[
                cls_ranking_df.score >= cls_thrs[i]
            ].to_list()
        else:
            cls_biases = cls_ranking_df.bias[cls_ranking_df.score > 0].to_list()[:top]

        all_biases += cls_biases

    return list(set(all_biases))


def get_random_biases(
    dataset, ending, num_biases=30, percent_biases=None, verbose=False, SEED=None
):
    if SEED is None:
        SEED = constants.SEED
    dir_path = os.path.join(constants.CACHE_PATH, "biases_v2", dataset)
    all_biases = []
    if isinstance(num_biases, list):
        for cls, num_cls_biases in zip(constants.DATASET_CLASSES[dataset], num_biases):
            cls_ranking_path = os.path.join(dir_path, f"{cls}_ranking{ending}.csv")
            cls_ranking_df = pd.read_csv(cls_ranking_path)

            cls_biases = cls_ranking_df.bias[cls_ranking_df.score > 0].to_list()
            if percent_biases is not None:
                np.random.seed(SEED)
                cls_biases = np.random.choice(
                    cls_biases, int(percent_biases * len(cls_biases)), replace=False
                ).tolist()
            else:
                np.random.seed(SEED)
                cls_biases = np.random.choice(
                    cls_biases, min(num_cls_biases, len(cls_biases)), replace=False
                ).tolist()
            all_biases += cls_biases
            if verbose:
                print(cls, num_cls_biases)
    else:
        for cls in constants.DATASET_CLASSES[dataset]:
            cls_ranking_path = os.path.join(dir_path, f"{cls}_ranking{ending}.csv")
            cls_ranking_df = pd.read_csv(cls_ranking_path)

            cls_biases = cls_ranking_df.bias[cls_ranking_df.score > 0].to_list()
            if percent_biases is not None:
                np.random.seed(SEED)
                cls_biases = np.random.choice(
                    cls_biases, int(percent_biases * len(cls_biases)), replace=False
                ).tolist()
            else:
                np.random.seed(SEED)
                cls_biases = np.random.choice(
                    cls_biases, min(num_biases, len(cls_biases)), replace=False
                ).tolist()
            all_biases += cls_biases

    return list(set(all_biases))


def debiasing_loss_l2(
    layer: torch.nn.Linear, biases: torch.Tensor, device: torch.device
):
    """
    layer - instance of Linear with row normalized weights
    biases - L2 normalized embeddings of biases to balance against
    """
    layer.to(device)
    biases = biases.to(device)

    scores = layer(biases)  # [N, C]

    thr = torch.mean(scores, axis=-1, keepdim=True).detach()
    loss = torch.mean(torch.square(scores - thr))

    return loss
