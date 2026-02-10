# Official repository for "Bridging Explainability and Embeddings: BEE Aware of Spuriousness" (ICLR 2026)

[\[Arxiv\]](https://arxiv.org/abs/2410.18970) \[Blog Post\]

**Authors**: Cristian Daniel Paduraru, Antonio Barbalau, Radu Filipescu, Andrei Liviu Nicolicioiu, Elena Burceanu 

**Abstract**: Current methods for detecting spurious correlations rely on analyzing dataset statistics or error patterns, leaving many harmful shortcuts invisible when counterexamples are absent. We introduce **BEE** (Bridging Explainability and Embeddings), a framework that shifts the focus from model predictions to the weight space, and to the embedding geometry underlying decisions. By analyzing how fine-tuning perturbs pretrained representations, BEE uncovers spurious correlations that remain hidden from conventional evaluation pipelines. We use linear probing as a transparent diagnostic lens, revealing spurious features that not only persist after full fine-tuning but also transfer across diverse state-of-the-art models. Our experiments cover numerous datasets and domains: vision (Waterbirds, CelebA, ImageNet-1k), language (CivilComments, MIMIC-CXR medical notes), and multiple embedding families (CLIP, CLIP-DataComp.XL, mGTE, BLIP2, SigLIP2). BEE consistently exposes spurious correlations: from concepts that slash the ImageNet accuracy by up to 95\%, to clinical shortcuts in MIMIC-CXR notes that induce dangerous false negatives. Together, these results position BEE as a general and principled tool for diagnosing spurious correlations in weight space, enabling principled dataset auditing and more trustworthy foundation models.

## Method Overview

![Figure 2](./images/Figure_Method.png "")

## Examples

![Figure 1](./images/Figure_ImageNet.png "")

![Table 2](./images/Table_Examples.png "")

![Table 3](./images/Table_Models.png "")


## Setup

With anaconda the environment used for development can be recreated using:

```
conda env create -f environment.yml
```


<!-- ## Data Collecton

Instructions for downloading and preprocessing data will be made available in the next update. -->

## Main Procedures

Cache the data embeddings and caption the dataset (only for the image classification ones).
```
python step_0_cache_embeddings_and_caption.py --dataset <dataset_name>
// Example
python step_0_cache_embeddings_and_caption.py --dataset Waterbirds
```

### Step 1
Perform ERM on the dataset to learn the SCs. Add the `--only_spurious` flag for the experiments in Sec 5.3, where only samples containing SCs are used. If no GPU is available, also pass the `--device` argument.
```
python step_1_ERM --dataset <dataset_name> [--device cpu] [--only_spurious]
// Example
python step_1_ERM --dataset Waterbirds --only_spurious
```
### Step 2
Extract keywords, filter out class related concepts, then rank the remaining ones and apply the dynamic threshold. Step 2a requires GPU for the LLM based filtering.
```
python step_2a_filter --dataset <dataset_name> [--only_spurious] 
python step_2bc_rank_and_threshold --dataset <dataset_name> [--only_spurious]

// Example
python step_2a_filter --dataset Waterbirds --only_spurious 
python step_2bc_rank_and_threshold --dataset Waterbirds --only_spurious
```

### Experiments from Section 5.2 - Training in a Fully Spurious Setup
SC regularization experiments
Perform linear probing with SC regularization (requires running steps 1&2 with the `--only_spurious` flag).

```
python bias_regularization.py --dataset <dataset_name>  --only_spuriouds [--random_biases]
// Example 
python bias_regularization.py --dataset CivilComments  --only_spuriouds --random_biases
```

GroupDRO only on the samples showcasing spuriously correlated attributes:
```
python lp_groupdro.py --dataset <dataset_name> --only_spurious
// Example
python lp_groupdro.py --dataset CelebA --only_spurious
```

## Citation

```
@inproceedings{paduraru2026bee,
    author       = {Cristian Daniel Paduraru and 
                    Antonio Barbalau and 
                    Radu Filipescu and 
                    Andrei Liviu Nicolicioiu and 
                    Elena Burceanu},
    title        = {Bridging Explainability and Embeddings: BEE Aware of Spuriousness},
    booktitle    = {{ICLR}},
    year         = {2026},
}
```

