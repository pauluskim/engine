import argparse
import time

from tqdm import tqdm

from learning_spoons.index.ls_hnsw import LSHnsw
from learning_spoons.data.utils import load_testcases
from learning_spoons.index.ls_faiss_index import LSFaiss
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
            self.index = LSVanilla(self.model)
        elif self.index_type == "faiss":
            self.index = LSFaiss(self.model)
        else:  # hnsw
            self.index = LSHnsw(self.model)
        self.index.load(self.index_path)

    def run(self):
        elapsed_times = []
        for _, row in tqdm(self.cases.iterrows(), desc="Evaluation", total=len(self.cases)):
            query_vector = self.model.infer(row["query"]).cpu()
            start = time.process_time()
            self.index.search(query_vector, 30)
            elapsed_times.append((time.process_time() - start) * 1000)
        print("\n[{}]\t\t Avg Elapsed Time: {:.2f}ms"
              .format(self.index_type, sum(elapsed_times) / len(elapsed_times)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--index_type", type=str)
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large")
    args = parser.parse_args()
    checker = LatencyChecker(args)
    checker.run()