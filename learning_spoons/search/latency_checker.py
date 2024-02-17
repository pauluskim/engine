import argparse
import pickle

import torch
from faiss import read_index
from tqdm import tqdm

from learning_spoons.data.utils import load_testcases
from learning_spoons.model.sentence_bert import SentenceBert


class LatencyChecker:
    def __init__(self, args):
        self.index_type = args.index_type
        self.index_path = args.index_path
        self.load_index()
        self.cases = load_testcases(args.cases_path)
        self.model = SentenceBert(model_name=args.model_name)

    def load_index(self):
        if self.index_type == "vanilla":
            self.index = torch.load(self.index_path, map_location=torch.device("cpu"))
        elif self.index_type == "faiss":
            self.index = read_index(self.index_path)
        else:  # hnsw
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)

    def run(self):
        for _, row in tqdm(self.cases.iterrows(), desc="Evaluation"):
            query_vector = self.model.infer(row["query"]).cpu()

            if index_type






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--index_type", type=str)
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large")
    args = parser.parse_args()
    checker = LatencyChecker(args)
    checker.run()