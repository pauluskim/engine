import argparse
import itertools
import os
import pdb

from data.ls_dataset import LSDataset
from data.utils import load_testcases
from index.ls_faiss_index import LSFaiss
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
                "delimiter": ["\n"],
                "grouping": [["idx", "title", "section"]],
                "section_weight": [
                    {"강사소개": 0.1},
                    {"강사소개": 1},
                    {"강사소개": 2}
                ]
            }
        }
        self.dataset_path = args.dataset
        self.cases = load_testcases(args.cases_path)
        self.index_root_path = args.index_root_path

    def eval(self, model_name, dataset_param):

        model = SentenceBert(model_name=model_name)
        dataset = LSDataset(self.dataset_path, dataset_param)

        inference = LSFaiss(model, dataset, batch_size=16)
        iter_name = f"{model_name}_{dataset_param}"
        index_fname = f"{iter_name}.index"
        index_fpath = os.path.join(self.index_root_path, index_fname)
        inference.indexing(index_fpath)

        evaluation = LSEvaluation(self.cases, model, dataset)
        result = evaluation.faiss(index_fpath)
        iter_result_name = f"{iter_name}_result.txt"
        avg_score = 1.0 * sum(result) / len(result)
        evaluation.cases
        df['new_col'] = mylist
        with open(os.path.join(self.index_root_path, iter_result_name), "w") as f:
          pdb.set_trace()
          evaluation.cases
          f.write(f"{result}")
        return avg_score


    def explore(self):
        dataset_params = self.params["dataset"]
        best_score = 0
        best_name = ""
        for model in self.params["st_model"]:
            keys, values = zip(*dataset_params.items())
            for dataset_param in [dict(zip(keys, v)) for v in itertools.product(*values)]:
                score = self.eval(model, dataset_param)
                if best_score < score:
                  best_score = score
                  best_name = f"{model}_{dataset_param}"
        with open(os.path.join(self.index_root_path, "final_result"), "w") as f:
          f.write(best_name + "\n")
          f.write(f"score: {best_score}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_root_path", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    gs = GridSearch(args)
    gs.explore()
