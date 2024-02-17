import argparse
import math

import faiss
import numpy as np
import torch
from faiss import write_index, read_index
from torch.nn import functional
from tqdm import tqdm

from data.ls_dataset import LSDataset
from data.utils import load_args, mkdir_if_not_exist
from learning_spoons.index.IndexInterface import IndexInterface
from model.sentence_bert import SentenceBert


class LSFaiss(IndexInterface):
    def prepare_index(self, dataset, batch_size, nprob_ratio=1.0):
        super(LSFaiss, self).prepare_index(dataset, batch_size)
        # This model's max_seq_length = 128

        # That means, the position embedding layer of the transformers has 512 weights,
        # but this sentence transformer will only use and was trained with the first 128 of them.
        # Therefore, you should be careful with increasing the value above 128.
        # It will work from a technical perspective,
        # but the position embedding weights (>128) are not properly trained and can therefore mess up your results.
        # Please also check this StackOverflow post.
        # https://stackoverflow.com/questions/75901231/max-seq-length-for-transformer-sentence-bert

        # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        # But the training vectors should be more than 30 * n_clusters.
        n_clusters = int(min(4 * math.sqrt(self.data_size), self.data_size / 40 - 1))

        vec_dimension = self.get_vector_dimenstion()

        # We use Inner Product (dot-product) as Index.
        # We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        # Need to init the FAISS index
        quantizer = faiss.IndexFlatIP(vec_dimension)
        self.index = faiss.IndexIVFFlat(quantizer, vec_dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
        # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.

        self.index.nprobe = int(n_clusters * nprob_ratio)

    def get_vector_dimenstion(self):
        _, _, context, _, _ = next(iter(self.data_loader))
        return self.model.infer(context).size()[1]

    def indexing(self, output_path):
        vector_lst = []

        counter = 0
        for lec_ids, lec_titles, docs, section, weights in tqdm(self.data_loader, desc="Index vectors"):
            # Need to get vector
            doc_vectors = functional.normalize(self.model.infer(docs), p=2.0, dim=1)
            vector_lst.append(doc_vectors)
            counter += 1

        # Need to aggregate vectors
        vectors = torch.cat(vector_lst, dim=0).cpu().numpy()
        # Need to index

        self.index.train(vectors)
        self.index.add(vectors)

        mkdir_if_not_exist(output_path)
        write_index(self.index, output_path)
        # Index * index = read_index("large.index")

    def load(self, index_path):
        self.index = read_index(index_path)

    def search(self, query_embedding, k):
        # expand dim for query vector
        query_embedding = np.expand_dims(query_embedding.cpu(), axis=0)
        scores, corpus_ids = self.index.search(query_embedding, k)
        return corpus_ids[0], scores[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    args = load_args(config_path)

    model = SentenceBert()
    dataset = LSDataset(args["dataset"], model.model.tokenizer)

    inference = LSFaiss(model, dataset, batch_size=16)
    index_fpath = args["index_output"]["faiss"]
    inference.indexing(index_fpath)