import argparse
import math

import faiss
import torch
from faiss import write_index
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.rating_dataset import RatingDataset
from data.utils import load_args, mkdir_if_not_exist
from model.sentence_bert import SentenceBert


class Faiss:
    def __init__(self, dataset, batch_size):
        self.model = SentenceBert()
        self.data_loader = DataLoader(dataset, batch_size=batch_size)
        self.data_size = len(dataset)

        # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        n_clusters = 4 * math.sqrt(self.data_size)

        vec_dimension = self.get_vector_dimenstion()

        # We use Inner Product (dot-product) as Index.
        # We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        quantizer = faiss.IndexFlatIP(vec_dimension)
        self.index = faiss.IndexIVFFlat(quantizer, vec_dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
        # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.

        self.index.nprobe = 3  # number of clusters to explorer at search time.

    def get_vector_dimenstion(self):
        doc_id, context = next(iter(self.data_loader))
        return self.model.infer(context).size()[1]

    def indexing(self, output_path):
        vector_lst = []
        for doc_ids, contexts in tqdm(self.data_loader, desc="Index vectors"):
            vector_lst.append(self.model.infer(contexts))

        vectors: Tensor = torch.cat(vector_lst, 0)
        faiss.normalize_L2(vectors)
        self.index.train(vectors)
        self.index.add(vectors)

        mkdir_if_not_exist(output_path)
        write_index(self.index, output_path)
        # Index * index = read_index("large.index")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    args = load_args(config_path)
    rating_dataset = RatingDataset(args["rating_dataset"])

    inference = Faiss(rating_dataset, batch_size=64)
    index_fpath = args["index_output"]["faiss"]
    inference.indexing(index_fpath)