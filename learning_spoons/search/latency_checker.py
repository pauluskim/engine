import argparse
import pickle

import torch
from faiss import read_index
from tqdm import tqdm

from learning_spoons.data.utils import load_testcases
from learning_spoons.index.ls_faiss_index import LSFaiss
from learning_spoons.index.ls_hnsw import LSHnsw
from learning_spoons.index.ls_vanilla import LSVanilla
from learning_spoons.model.sentence_bert import SentenceBert


class LatencyChecker:
    def __init__(self, args):
        self.index_type = args.index_type
        self.index_path = args.index_path
        self.cases = load_testcases(args.cases_path)
        self.model = SentenceBert(model_name=args.model_name)
        self.load_index()

    def load_index(self):
        if self.index_type == "vanilla":
            self.index = LSVanilla(self.model, None, 1)
        elif self.index_type == "faiss":
            self.index = LSFaiss(self.model, None, 1)
        else:  # hnsw
            self.index = LSHnsw(self.model, None, 1, None, None)
        self.index.load(self.index_path)

    def run(self):
        for _, row in tqdm(self.cases.iterrows(), desc="Evaluation"):
            query_vector = self.model.infer(row["query"]).cpu()
            self.index.search(query_vector, 30)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--index_type", type=str)
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large")
    args = parser.parse_args()
    checker = LatencyChecker(args)
    checker.run()