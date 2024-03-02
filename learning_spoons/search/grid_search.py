import argparse
import os

import pandas as pd

from learning_spoons.data.ls_dataset import LSDataset
from learning_spoons.data.utils import load_testcases, load_args
from learning_spoons.index.ls_faiss_index import LSFaiss
from learning_spoons.index.ls_vanilla import LSVanilla
from learning_spoons.model.sentence_bert import SentenceBert
from learning_spoons.search.ls_evaluation import LSEvaluation


class GridSearch:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.index_type = args.index_type
        self.params = load_args(args.grid_param_path)
        self.index_root_path = args.index_root_path
        self.batch_size = 16
        self.skip_index = args.skip_index
        self.cases = load_testcases(args.cases_path)

    def eval(self, index, dataset, index_path, result_path, k):
        if self.skip_index and os.path.isfile(index_path):
            print("SKIP to index: " + index_path)
        else:
            inference.indexing(index_fpath)

        if self.skip_index and os.path.isfile(result_path):
            print("SKIP to evaluate: " + result_path)
            df = pd.read_csv(result_path)
            return df["avg_score"][0]
        else:
            evaluation = LSEvaluation(self.cases, self.model, dataset, k)
            index.load(index_path)
            return evaluation.main(index, result_path)


    def explore(self):
        dataset_params = self.params["dataset"]
        result_lst = []
        # 모델별로 report를 작성하기 위한 for-loop
        #   each iteration에서 나머지 dataset_params의 조합으로 실행 및 저장
        #   index type 별로 index를 instance하여 실행


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--grid_param_path", type=str)
    parser.add_argument("--index_root_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--index_type", type=str)
    parser.add_argument("--skip_index", type=bool, default=False)
    args = parser.parse_args()
    gs = GridSearch(args)
    gs.explore()
