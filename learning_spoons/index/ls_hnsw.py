import argparse
import sys

from data.utils import load_args, mkdir_if_not_exist

# sys.path.append("/content/drive/MyDrive/colab/engine")


import pickle

import hnswlib
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.rating_dataset import RatingDataset
from learning_spoons.index.IndexInterface import IndexInterface
from model.sentence_bert import SentenceBert


class LSHnsw(IndexInterface):
    def __init__(self, model, dataset, batch_size, ef, M):
        super(LSHnsw, self).__init__(model, dataset, batch_size)
        self.ef = ef
        self.m = M
        # https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        self.prepare_index()

    def prepare_index(self):
        # Need to init index
        test_vector = self.model.infer("test")
        vector_dim = test_vector.size()[0]
        self.p = hnswlib.Index(space='cosine', dim=vector_dim)

        self.p.init_index(max_elements=self.data_size, ef_construction=self.ef * 2, M=self.m)

    def indexing(self, output_path):
        counter = 0
        for lec_ids, lec_titles, docs, section, weights in tqdm(self.data_loader, desc="Index vectors"):

            # Need to add vector into the hnsw index
            vectors = self.model.infer(docs)
            self.p.add_items(vectors.cpu())
            counter += 1
            if counter == 10:
                break

        self.p.set_ef(self.ef)
        self.save(output_path)

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

    inference = LSHnsw(rating_dataset, batch_size=64)
    index_fpath = args["index_output"]["hnsw"]
    inference.indexing(index_fpath)