import argparse
import sys

from data.utils import load_args, mkdir_if_not_exist

# sys.path.append("/content/drive/MyDrive/colab/engine")


import pickle

import hnswlib
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.rating_dataset import RatingDataset
from model.sentence_bert import SentenceBert


class Hnsw():
    def __init__(self, dataset, batch_size):
        self.model = SentenceBert()
        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.data_size = len(dataset)

        self.prepare_index()

    def prepare_index(self):
        # Need to init index
        pass

    def indexing(self, output_path):
        for doc_ids, contexts in tqdm(self.data_loader, desc="Index vectors"):
            # Need to add vector into the hnsw index
            pass

        # Need to tune index and save
        pass

    def save(self, output_path):
        mkdir_if_not_exist(output_path)
        with open(output_path, "wb") as f:
            pickle.dump(self.p, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    args = load_args(config_path)
    rating_dataset = RatingDataset(args["rating_dataset"])

    inference = Hnsw(rating_dataset, batch_size=64)
    index_fpath = args["index_output"]["hnsw"]
    inference.indexing(index_fpath)