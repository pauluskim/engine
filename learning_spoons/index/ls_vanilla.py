import argparse

import torch
from torch.nn import functional
from tqdm import tqdm

from data.rating_dataset import RatingDataset
from data.utils import load_args, mkdir_if_not_exist
from index.IndexInterface import IndexInterface


class LSVanilla(IndexInterface):
    def indexing(self, output_path):
        vector_lst = []
        counter = 0
        for lec_ids, lec_titles, docs, section, weights in tqdm(self.data_loader, desc="Index vectors"):
            # need to infer and aggregate vectors
            doc_vectors = functional.normalize(self.model.infer(docs), p=2.0, dim =1)
            vector_lst.append(doc_vectors)
            counter += 1
            if counter == 5:
                break


        # need to convert from list to tensor
        vectors = torch.cat(vector_lst, dim=0)
        mkdir_if_not_exist(output_path)
        torch.save(vectors, output_path)

    def load(self, index_path):
        self.index = torch.load(index_path, map_location=torch.device("cpu"))

    def search(self, query_embedding, k):
        # Already query_embedding and corpus_embeddings are normalized.
        cos_scores = torch.inner(query_embedding, self.index)

        # Find the top k docs from the calculation results.
        scores, doc_idxs = torch.topk(cos_scores, k=k)
        return doc_idxs.cpu(), scores.cpu()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    args = load_args(config_path)
    rating_dataset = RatingDataset(args["rating_dataset"])

    inference = LSVanilla(rating_dataset, batch_size=64)
    index_fpath = args["index_output"]["vanilla"]
    inference.indexing(index_fpath)