TEMPLATES = {
    "Waterbirds": ["a photo of a {cls} in the {bias}"],
    "CelebA": ["a photo of a {bias} with {cls}"],
    "CivilComments": ["a/an {cls} comment about {bias}"],
    "NICOpp": ["a photo of a {cls} in the {bias}"],
}

BASE_TEMPLATE = {
    "Waterbirds": "a photo of a {cls}",
    "CelebA": "a photo of a person with {cls}",
    "CivilComments": "{cls}",
    "NICOpp": "a photo of a {cls}",
}
