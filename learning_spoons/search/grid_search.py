import argparse
import itertools
import os

import pandas as pd

from learning_spoons.data.ls_dataset import LSDataset
from learning_spoons.data.utils import load_testcases
from learning_spoons.index.ls_faiss_index import LSFaiss
from learning_spoons.index.ls_vanilla import LSVanilla
from learning_spoons.model.sentence_bert import SentenceBert
from learning_spoons.search.ls_evaluation import LSEvaluation


class GridSearch:
    def __init__(self, args):
        self.params = {
            "st_model": ["jhgan/ko-sroberta-multitask"],
            "dataset": {
                "delimiter": [" "],
                "grouping": [None, ["idx", "title", "section"], ["idx", "title"]],
                "section_weight": [
                    {"강사소개": 0.1, "title": 1, "강의소개": 1, "인트로": 1},
                ],
                "retrieval_candidate_times": [15]
            }
        }
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
        self.index_root_path = args.index_root_path
        self.index_type = args.index_type
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
            score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst = (
                evaluation.vanilla(index_fpath))
            avg_score = 1.0 * sum(score_lst) / len(score_lst)
            evaluation.cases['recall'] = score_lst
            evaluation.cases['retrieved_docs'] = retrieved_docs_lst
            evaluation.cases['expected_details'] = expected_lec_detail_lst
            evaluation.cases['result_details'] = search_result_detail_lst
            evaluation.cases['avg_score'] = avg_score
            evaluation.cases.to_csv(iter_result_path)
            return avg_score

    def eval_by_faiss_index(self, model_name, dataset_param):

        model = SentenceBert(model_name=model_name)
        dataset = LSDataset(self.dataset_path, dataset_param)

        inference = LSFaiss(model, dataset, batch_size=16)
        iter_name = f"{model_name}_{dataset_param}"
        index_fname = f"{iter_name}.index"
        index_fpath = os.path.join(self.index_root_path, index_fname)
        if self.skip_index is False:
            inference.indexing(index_fpath)

        evaluation = LSEvaluation(self.cases, model, dataset, dataset_param["retrieval_candidate_times"])
        score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst = evaluation.faiss(index_fpath)
        iter_result_name = f"{iter_name}_result.csv"
        avg_score = 1.0 * sum(score_lst) / len(score_lst)
        evaluation.cases['recall'] = score_lst
        evaluation.cases['retrieved_docs'] = retrieved_docs_lst
        evaluation.cases['expected_details'] = expected_lec_detail_lst
        evaluation.cases['result_details'] = search_result_detail_lst
        evaluation.cases.to_csv(os.path.join(self.index_root_path, iter_result_name))
        return avg_score

    def explore(self):
        dataset_params = self.params["dataset"]
        result_lst = []
        for model_name in self.params["st_model"]:
            print("MODEL: ", model_name)
            self.model_name = model_name
            self.model = SentenceBert(model_name=self.model_name)

            keys, values = zip(*dataset_params.items())
            for dataset_param in [dict(zip(keys, v)) for v in itertools.product(*values)]:
                if self.index_type == "faiss":
                    score = self.eval_by_faiss_index(dataset_param)
                else:
                    score = self.eval_by_vanilla_index(dataset_param)
                result_lst.append([str(score), f"{model_name}_{dataset_param}"])

            with open(os.path.join(self.index_root_path, model_name + "_final_result.csv"), "w") as f:
                for result in result_lst:
                    f.write("\t".join(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_root_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--index_type", type=str)
    parser.add_argument("--skip_index", type=bool, default=False)
    args = parser.parse_args()
    gs = GridSearch(args)
    gs.explore()
