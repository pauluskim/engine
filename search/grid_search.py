import argparse
import itertools
import os
import pdb


from data.ls_dataset import LSDataset
from data.utils import load_testcases
from index.ls_faiss_index import LSFaiss
from index.ls_vanilla import LSVanilla
from model.sentence_bert import SentenceBert
from search.ls_evaluation import LSEvaluation


class GridSearch:
    def __init__(self, args):
        """
            "delimiter": [" ", "\n"], # newline, space 차이가 없음
            "grouping": [None, ["idx", "title", "section"], ["idx", "title"]],  idx, title, section이 가장 좋음
        """
        self.params = {
            "st_model": ["jhgan/ko-sroberta-multitask"],
            "dataset": {
                "delimiter": [" "],
                "grouping": [["idx", "title", "section"]],
                "section_weight": [
                    {"강사소개": 0.1}
                ]
            }
        }
        self.dataset_path = args.dataset
        self.cases = load_testcases(args.cases_path)
        self.index_root_path = args.index_root_path
        self.index_type = args.index_type

    def vanilla_eval(self, model_name, dataset_param):

        model = SentenceBert(model_name=model_name)
        dataset = LSDataset(self.dataset_path, dataset_param)

        inference = LSVanilla(dataset, 16)
        iter_name = f"{model_name}_{dataset_param}"
        index_fname = f"{iter_name}.pth"
        index_fpath = os.path.join(self.index_root_path, index_fname)
        inference.indexing(index_fpath)

        evaluation = LSEvaluation(self.cases, model, dataset)
        score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst = evaluation.vanilla(index_fpath)
        iter_result_name = f"{iter_name}_vanilla_result.csv"
        avg_score = 1.0 * sum(score_lst) / len(score_lst)
        evaluation.cases['recall'] = score_lst
        evaluation.cases['retrieved_docs'] = retrieved_docs_lst
        evaluation.cases['expected_details'] = expected_lec_detail_lst
        evaluation.cases['result_details'] = search_result_detail_lst
        evaluation.cases.to_csv(os.path.join(self.index_root_path, iter_result_name))
        return avg_score

    def faiss_eval(self, model_name, dataset_param):

        model = SentenceBert(model_name=model_name)
        dataset = LSDataset(self.dataset_path, dataset_param)

        inference = LSFaiss(model, dataset, batch_size=16)
        iter_name = f"{model_name}_{dataset_param}"
        index_fname = f"{iter_name}.index"
        index_fpath = os.path.join(self.index_root_path, index_fname)
        inference.indexing(index_fpath)

        evaluation = LSEvaluation(self.cases, model, dataset)
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
        for model in self.params["st_model"]:
            keys, values = zip(*dataset_params.items())
            for dataset_param in [dict(zip(keys, v)) for v in itertools.product(*values)]:
                if self.index_type == "faiss":
                    score = self.faiss_eval(model, dataset_param)
                else:
                    score = self.vanilla_eval(model, dataset_param)
                result_lst.append([str(score), f"{model}_{dataset_param}"])

        with open(os.path.join(self.index_root_path, "final_result.csv"), "w") as f:
            for result in result_lst:
                f.write("\t".join(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_root_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--index_type", type=str)
    args = parser.parse_args()
    gs = GridSearch(args)
    gs.explore()
