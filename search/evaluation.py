import argparse
import os
import pickle
import time

import numpy as np
import torch
import yaml
from faiss import read_index
from sentence_transformers import util

from model.sentence_bert import SentenceBert


class Evaluation:
    def __init__(self, env='local', verbose=True, top_k=5):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        fpath = os.path.join(dir_path, f"../data/{env}_config.yaml")
        with open(fpath) as f:
            self.path_config = yaml.load(f, Loader=yaml.FullLoader)
        self.model = SentenceBert()
        self.queries = ["재밌는 영화", "여주인공이 예뻣던 영화", "킬링 타임용 영화", "답답해서 암걸릴 것 같은 영화"]
        self.load_reviews()
        self.k = top_k
        self.verbose = verbose

    def load_reviews(self):
        self.idx2review = []
        with open(self.path_config["rating_dataset"], "r") as f:
            f.readline()
            for line in f:
                review = line.strip().split("\t")[1]
                self.idx2review.append(review)

    def vanilla(self):
        corpus_embeddings = torch.load(self.path_config["index_output"]["vanilla"], map_location=torch.device('cpu'))

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(self.k, len(corpus_embeddings))
        elapsed_times = []
        for query in self.queries:
            query_embedding = self.model.infer(query)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            start = time.process_time()
            # Search with the corpus embeddings
            # 1. Need to calculcate cos_sim
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            # 2. Find the top k docs from the calculation results.
            top_results = torch.topk(cos_scores, k=top_k)
            elapsed_times.append((time.process_time()- start) * 1000)

            self.print_search_result(query, top_results[0], top_results[1])

        print("\n[VANILLA]\t\t Avg Elapsed Time: {:.2f}ms".format(sum(elapsed_times) / len(elapsed_times)))

    def faiss(self):
        index = read_index(self.path_config["index_output"]["faiss"])

        elapsed_times = []
        for query in self.queries:
            query_vector = self.model.infer(query)
            # expand dim for query vector
            query_vectory = np.expand_dims(query_vector, axis=0)
            start = time.process_time()
            distances, corpus_ids = index.search(query_vectory, self.k)
            elapsed_times.append((time.process_time()- start) * 1000)

            self.print_search_result(query, distances[0], corpus_ids[0])
        print("\n[FAISS]\t\t Avg Elapsed Time: {:.2f}ms".format(sum(elapsed_times) / len(elapsed_times)))

    def hnsw(self):
        with open(self.path_config["index_output"]["hnsw"], "rb") as f:
            corpus_embeddings = pickle.load(f)

        elapsed_times = []
        for query in self.queries:
            query_vector = self.model.infer(query)

            start = time.process_time()
            doc_ids, distances = corpus_embeddings.knn_query(query_vector, k=self.k)
            elapsed_times.append((time.process_time()- start) * 1000)
            self.print_search_result(query, distances[0], doc_ids[0])
        print("\n[HNSW]\t\t Avg Elapsed Time: {:.2f}ms".format(sum(elapsed_times) / len(elapsed_times)))

    def print_search_result(self, query, scores, doc_idxs):
        if self.verbose:
            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 5 most similar sentences in corpus:")

            for score, idx in zip(scores, doc_idxs):
                print(self.idx2review[idx], "(Score: {:.4f})".format(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    args = parser.parse_args()
    eval = Evaluation(env=args.env, verbose=False)
    eval.vanilla()
    eval.hnsw()
    eval.faiss()
