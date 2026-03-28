from datasets import load_dataset


def load_sst2_dataset():
    dataset = load_dataset("glue", "sst2")
    return dataset
