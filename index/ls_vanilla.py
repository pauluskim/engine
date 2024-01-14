import argparse
import pdb

import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.rating_dataset import RatingDataset
from data.utils import load_args, mkdir_if_not_exist
from index.IndexInterface import IndexInterface
from model.sentence_bert import SentenceBert


class LSVanilla(IndexInterface):
    def indexing(self, output_path):
        vector_lst = []
        counter = 0
        for doc_ids, lec_ids, lec_titles, docs, section, weights in tqdm(self.data_loader, desc="Index vectors"):
            # need to infer and aggregate vectors
            doc_vectors = functional.normalize(self.model.infer(docs), p=2.0, dim =1)
            vector_lst.append(doc_vectors)
            counter += 1


        # need to convert from list to tensor
        vectors = torch.cat(vector_lst, dim=0)
        mkdir_if_not_exist(output_path)
        # need tosave the embedding vectors
        torch.save(vectors, output_path)

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