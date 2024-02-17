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
        # self.params = {
        #     "st_model": ["jhgan/ko-sroberta-multitask",
        #                  "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        #                  "intfloat/multilingual-e5-large",
        #                  "jhgan/ko-sbert-sts"],
        #     "dataset": {
        #         "delimiter": [" ", "\n"],
        #         "grouping": [None, ["idx", "title", "section"], ["idx", "title"]],
        #         "section_weight": [
        #             {"강사소개": 0.1, "title": 1, "강의소개": 1, "인트로": 1},
        #             {"강사소개": 0.1, "title": 2, "강의소개": 1, "인트로": 1},
        #             {"강사소개": 0.1, "title": 1, "강의소개": 2, "인트로": 1},
        #             {"강사소개": 0.1, "title": 2, "강의소개": 2, "인트로": 1},
        #             {"강사소개": 0.1, "title": 1, "강의소개": 1, "인트로": 2},
        #             {"강사소개": 0.1, "title": 2, "강의소개": 1, "인트로": 2},
        #             {"강사소개": 0.1, "title": 1, "강의소개": 2, "인트로": 2},
        #             {"강사소개": 0.1, "title": 2, "강의소개": 2, "인트로": 2},
        #         ],
        #         "retrieval_candidate_times": [15, 30, 50]
        #     }
        # }
        self.dataset_path = args.dataset_path
        self.index_type = args.index_type
        self.params = load_args(args.grid_param_path)
        self.index_root_path = args.index_root_path
        self.batch_size = 16
        self.skip_index = args.skip_index
        self.cases = load_testcases(args.cases_path)

    def eval_by_vanilla_index(self, dataset_param):
        dataset = LSDataset(self.dataset_path, dataset_param)
        index = LSVanilla(self.model, dataset, self.batch_size)

        iter_name = f"{self.model_name}_{dataset_param}"
        index_fname = f"{iter_name}.pth"
        index_fpath = os.path.join(self.index_root_path, index_fname)

        if self.skip_index and os.path.isfile(index_fpath):
            print("SKIP to index: " + index_fpath)
        else:
            index.indexing(index_fpath)

        iter_result_name = f"{iter_name}_vanilla_result.csv"
        iter_result_path = os.path.join(self.index_root_path, iter_result_name)

        if self.skip_index and os.path.isfile(iter_result_path):
            print("SKIP to evaluate: " + iter_result_name)
            df = pd.read_csv(iter_result_path)
            return df["avg_score"][0]
        else:
            evaluation = LSEvaluation(self.cases, self.model, dataset, dataset_param["retrieval_candidate_times"])
            return evaluation.vanilla(index_fpath, iter_result_path)

    def eval_by_faiss_index(self, dataset_param):
        dataset = LSDataset(self.dataset_path, dataset_param)
        inference = LSFaiss(self.model, dataset, self.batch_size, dataset_param.get("faiss_nprob_ratio", 1.0))

        iter_name = f"{self.model_name}_{dataset_param}"
        index_fname = f"{iter_name}.index"
        index_fpath = os.path.join(self.index_root_path, index_fname)

        if self.skip_index and os.path.isfile(index_fpath):
            print("SKIP to index: " + index_fpath)
        else:
            inference.indexing(index_fpath)

        iter_result_name = f"{iter_name}_faiss_result.csv"
        iter_result_path = os.path.join(self.index_root_path, iter_result_name)

        if self.skip_index and os.path.isfile(iter_result_path):
            print("SKIP to evaluate: " + iter_result_name)
            df = pd.read_csv(iter_result_path)
            return df["avg_score"][0]
        else:
            evaluation = LSEvaluation(self.cases, self.model, dataset, dataset_param["retrieval_candidate_times"])
            return evaluation.faiss(index_fpath, iter_result_path)

    def eval_by_hnsw_index(self, dataset_param):
        dataset = LSDataset(self.dataset_path, dataset_param)
        # TODO: 5 to 1.
        k_cap = 5 * dataset_param["retrieval_candidate_times"]
        inference = LSHnsw(self.model, dataset, self.batch_size, k_cap, dataset_param.get("M", 48))

        iter_name = f"{self.model_name}_{dataset_param}"
        index_fname = f"{iter_name}.pickle"
        index_fpath = os.path.join(self.index_root_path, index_fname)

        if self.skip_index and os.path.isfile(index_fpath):
            print("SKIP to index: " + index_fpath)
        else:
            inference.indexing(index_fpath)

        iter_result_name = f"{iter_name}_hnsw_result.csv"
        iter_result_path = os.path.join(self.index_root_path, iter_result_name)

        if self.skip_index and os.path.isfile(iter_result_path):
            print("SKIP to evaluate: " + iter_result_name)
            df = pd.read_csv(iter_result_path)
            return df["avg_score"][0]
        else:
            evaluation = LSEvaluation(self.cases, self.model, dataset, dataset_param["retrieval_candidate_times"])
            return evaluation.hnsw(index_fpath, iter_result_path)

    def explore(self):
        dataset_params = self.params["dataset"]
        for model_name in self.params["st_model"]:
            result_lst = []
            print("MODEL: ", model_name)
            self.model_name = model_name
            self.model = SentenceBert(model_name=self.model_name)

            keys, values = zip(*dataset_params.items())
            for dataset_param in [dict(zip(keys, v)) for v in itertools.product(*values)]:
                if self.index_type == "faiss":
                    score = self.eval_by_faiss_index(dataset_param)
                elif self.index_type == "hnsw":
                    score = self.eval_by_hnsw_index(dataset_param)
                else:
                    score = self.eval_by_vanilla_index(dataset_param)
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
