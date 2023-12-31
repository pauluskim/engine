import argparse
import itertools
import os

from data.ls_dataset import LSDataset
from data.utils import load_testcases
from index.ls_faiss_index import LSFaiss
from model.sentence_bert import SentenceBert
from search.ls_evaluation import LSEvaluation


class GridSearch:
    def __init__(self, args):
        self.params = {
            "st_model": ["jhgan/ko-sroberta-multitask"],
            "dataset": {
                "delimiter": [" ", "\n"],
                "grouping": [None, ["idx", "title", "section"], ["idx", "title"]],
                "section_weight": [
                    {"강사소개": 0.1}
                ]
            }
        }
        self.dataset_path = args.dataset
        self.cases = load_testcases(args.cases_path)
        self.index_root_path = args.index_root_path

    def eval(self, model, dataset_param):

        model = SentenceBert(model_name=model)
        dataset = LSDataset(self.dataset_path, dataset_param)

        inference = LSFaiss(model, dataset, batch_size=16)
        index_fname = f"{model}_{dataset_param.values()}.index"
        index_fpath = os.path.join(self.index_root_path, index_fname)
        inference.indexing(index_fpath)

        evaluation = LSEvaluation(self.cases, model, dataset)
        evaluation.faiss(args.faiss_index)

    def explore(self):
        dataset_params = self.params["dataset"]
        for model in self.params["st_model"]:
            keys, values = zip(*dataset_params.items())
            for dataset_param in [dict(zip(keys, v)) for v in itertools.product(*values)]:
                self.eval(model, dataset_param)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_root_path", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    gs = GridSearch(args)
    gs.explore()
