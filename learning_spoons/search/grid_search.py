import argparse
import itertools
import os

import pandas as pd

from learning_spoons.data.ls_dataset import LSDataset
from learning_spoons.data.utils import load_testcases, load_args
from learning_spoons.index.ls_faiss_index import LSFaiss
from learning_spoons.index.ls_hnsw import LSHnsw
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
            index.indexing(index_path)

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
        for model_name in self.params["st_model"]:
            result_lst = []
            print("MODEL: ", model_name)
            self.model_name = model_name
            self.model = SentenceBert(model_name=self.model_name)

            keys, values = zip(*dataset_params.items())
            for dataset_param in [dict(zip(keys, v)) for v in itertools.product(*values)]:
                dataset = LSDataset(self.dataset_path, dataset_param)
                iter_name = f"{self.model_name}_{dataset_param}"
                k = dataset_param["retrieval_candidate_times"]
                if self.index_type == "faiss":
                    index = LSFaiss(self.model, dataset, self.batch_size, dataset_param.get("faiss_nprob_ratio", 1.0))
                    index_path = os.path.join(self.index_root_path, f"{iter_name}.index")
                    result_path = os.path.join(self.index_root_path, f"{iter_name}_faiss_result.csv")
                    score = self.eval(index, dataset, index_path, result_path, k)
                elif self.index_type == "hnsw":
                    # TODO: 5 to 1 for k
                    index = LSHnsw(self.model, dataset, self.batch_size, 5 * k, dataset_param.get("M", 48))
                    index_path = os.path.join(self.index_root_path, f"{iter_name}.pickle")
                    result_path = os.path.join(self.index_root_path, f"{iter_name}_hnsw_result.csv")
                    score = self.eval(index, dataset, index_path, result_path, k)
                else:
                    index = LSVanilla(self.model, dataset, self.batch_size)
                    index_path = os.path.join(self.index_root_path, f"{iter_name}.pth")
                    result_path = os.path.join(self.index_root_path, f"{iter_name}_vanilla_result.csv")
                    score = self.eval(index, dataset, index_path, result_path, k)
                result_lst.append([str(score), f"{model_name}_{dataset_param}"])

            with open(os.path.join(self.index_root_path, model_name + f"_{self.index_type}_final_result.csv"), "w") as f:
                for result in result_lst:
                    f.write("\t".join(result) + "\n")


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
