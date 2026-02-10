import os

DATA_PATH = os.environ.get("DATA_PATH", "../data/")
RESULTS_PATH = "../results/xai/"
CACHE_PATH = "../cache/xai/"
CKPT_PATH = "../ckpt/xai/"

METADATA_NAME = {
    "Waterbirds": "metadata_waterbirds.csv",
    "CelebA": "metadata_celeba.csv",
    "CivilComments": "metadata_civilcomments_coarse.csv",
}

DATASET_DIR = {
    "Waterbirds": "waterbirds",
    "CelebA": "celeba",
    "CivilComments": "civilcomments",
}

DATASET_IMAGE_DIR = {
    "Waterbirds": "waterbird_complete95_forest2water2",
    "CelebA": "img_align_celeba",
    "CivilComments": "",
}


DATASET_CLASSES = {
    "Waterbirds": ["landbird", "waterbird"],
    "CelebA": ["non-blonde hair", "blonde hair"],  # attr_0=female; attr_1=male
    "CivilComments": ["non-offensive", "offensive"],
}

DATASET_CLASS_BIAS = {
    "Waterbirds": [0, 1],  # 0=land, 1=water
    "CelebA": [1, 0],  # attr_0=female; attr_1=male
    "CivilComments": [1, 0],  # 0=minority mentioned, 1=no minority mentioned
}

DATASET_CLASSES_WN = {
    "Waterbirds": ["bird", "bird"],
    "CelebA": ["hair", "hair"],  # attr_0=female; attr_1=male
    "CivilComments": ["non-offensive", "offensive"],
}


DATASET_CLASSES_EXPLICIT = {
    "Waterbirds": "birds and any specific species of birds",
    "CelebA": "hair and its color (blonde, black, brown, red, gray etc.)",
    "CivilComments": "remove any offensive words or remarks",
}


DATASET_SPLITS = {"train": 0, "val": 1, "test": 2}

SPLITS = ["train", "val", "test"]

TEMP_INIT = 4.60517025

SEED = 1007  # 1013; 1019;

STOPWORDS = {"next", "many"}

# If nothing is left from the sequence or the remaining words do not hold any meaning then you must simply write None.
LLAMA_PROMPT = """I will provide a list of concepts and sequence of words. Your task is to remove any instance of the concepts from the given sequence. 
If no instance of any concept is present then you must return the sequence as is. 
Here are a few examples:
Example 1:
Concepts: [dogs and any specific species of dogs]
Sequence: 'a golden retriever with a bone'
Answer: 'bone'

Example 2:
Concepts: [clothing and anything related to their color]
Sequence: 'a shiny black and white dress'
Answer: 'shiny'

Example 3:
Concepts: [mentions of people\'s names]
Sequence: 'John is an assistant'
Answer: 'assistant'

Example 4:
Concepts: [cats, horses, dolls, the sun and any specific species or types of these concepts]
Sequence: 'A picture of the rising sun'
Answer: 'picture'

Now complete the following case, without thinking step by step or asking for anything else. 
Concepts: [{}]
Sequence: '{}'
Answer: """
