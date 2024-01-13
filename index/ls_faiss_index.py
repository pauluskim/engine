import argparse
import math

import faiss
import torch
from faiss import write_index
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ls_dataset import LSDataset
from data.utils import load_args, mkdir_if_not_exist
from index.IndexInterface import IndexInterface
from model.sentence_bert import SentenceBert


class LSFaiss(IndexInterface):
    def __init__(self, model, dataset, batch_size, nprob=50):
        super(model, dataset, batch_size)

        # This model's max_seq_length = 128

        # That means, the position embedding layer of the transformers has 512 weights,
        # but this sentence transformer will only use and was trained with the first 128 of them.
        # Therefore, you should be careful with increasing the value above 128.
        # It will work from a technical perspective,
        # but the position embedding weights (>128) are not properly trained and can therefore mess up your results.
        # Please also check this StackOverflow post.
        # https://stackoverflow.com/questions/75901231/max-seq-length-for-transformer-sentence-bert

        # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        # TODO: But the training vectors should be more than 30 * n_clusters.
        n_clusters = int(min(4 * math.sqrt(self.data_size), self.data_size / 30 - 1))

        vec_dimension = self.get_vector_dimenstion()

        # We use Inner Product (dot-product) as Index.
        # We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        # Need to init the FAISS index
        quantizer = faiss.IndexFlatIP(vec_dimension)
        self.index = faiss.IndexIVFFlat(quantizer, vec_dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
        # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.

        self.index.nprobe = nprob

    def get_vector_dimenstion(self):
        _, _, _, context, _, _ = next(iter(self.data_loader))
        return self.model.infer(context).size()[1]

    def indexing(self, output_path):
        vector_lst = []

        counter = 0
        for doc_ids, lec_ids, lec_titles, docs, section, weights in tqdm(self.data_loader, desc="Index vectors"):
            # Need to get vector
            # TODO: NEED TO VECTORIZE TITLE AS WELL.
            vector_lst.append(self.model.infer(docs))
            counter += 1

        # Need to aggregate vectors
        vectors = torch.cat(vector_lst, dim=0).cpu().numpy()
        # Need to index
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

    model = SentenceBert()
    dataset = LSDataset(args["dataset"], model.model.tokenizer)

    inference = LSFaiss(model, dataset, batch_size=16)
    index_fpath = args["index_output"]["faiss"]
    inference.indexing(index_fpath)