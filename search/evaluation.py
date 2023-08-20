import argparse
import os

import torch
import yaml
from sentence_transformers import util

from model.sentence_bert import SentenceBert


class Evaluation:
    def __init__(self, env='local', top_k=5):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        fpath = os.path.join(dir_path, f"../data/{env}_config.yaml")
        with open(fpath) as f:
            self.path_config = yaml.load(f, Loader=yaml.FullLoader)
        self.model = SentenceBert()
        self.queries = ["재밌는 영화", "여주인공이 예뻣던 영화"]
        self.reviews = self.load_reviews()
        self.k = top_k

    def laod_reviews(self):
        with open(self.path_config["rating_datset"], "r") as f:
            for line in f:


    def vanilla(self):
        corpus_embeddings = torch.load(self.path_config["index_output"]["vanilla"])

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(5, len(corpus_embeddings))
        for query in self.queries:
            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 5 most similar sentences in corpus:")

            for score, idx in zip(top_results[0], top_results[1]):
                print(corpus[idx], "(Score: {:.4f})".format(score))

    def vanilla_with_gpu(self):
        corpus_embeddings = corpus_embeddings.to('cuda')
        corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

        query_embeddings = query_embeddings.to('cuda')
        query_embeddings = util.normalize_embeddings(query_embeddings)
        hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)

    def faiss(self):
        pass

    def hnsw(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    args = parser.parse_args()
    eval = Evaluation(env=args.env)
    eval.vanilla()
