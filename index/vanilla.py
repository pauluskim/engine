import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.rating_dataset import RatingDataset
from data.utils import load_args, mkdir_if_not_exist
from model.sentence_bert import SentenceBert


class Vanilla:
    def __init__(self, dataset, batch_size):
        self.model = SentenceBert()
        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.data_size = len(dataset)

    def indexing(self, output_path):
        vector_lst = []
        for doc_ids, contexts in tqdm(self.data_loader, desc="Index vectors"):
            # need to infer and aggregate vectors
            pass

        # need to convert from list to tensor
        mkdir_if_not_exist(output_path)
        # need tosave the embedding vectors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    args = load_args(config_path)
    rating_dataset = RatingDataset(args["rating_dataset"])

    inference = Vanilla(rating_dataset, batch_size=64)
    index_fpath = args["index_output"]["vanilla"]
    inference.indexing(index_fpath)