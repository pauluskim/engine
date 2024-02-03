import argparse
import itertools
import os
import pdb

import pandas as pd

from data.ls_dataset import LSDataset
from data.utils import load_testcases
from index.ls_faiss_index import LSFaiss
from index.ls_vanilla import LSVanilla
from model.sentence_bert import SentenceBert
from search.ls_evaluation import LSEvaluation


class GridSearch:
    def __init__(self, args):
        """

            기본적으로 EBR은 2 tower 모델로 가는게 좋음
            https://medium.com/better-ml/embedding-learning-for-retrieval-29af1c9a1e65
            query tower
            document tower를 따로 가져가고, feed log, click 로그와 같은 걸로 학습시키는것이 좋음
            하지만 우리는 해당 데이터가 없기때문에, 오픈되어있는 범용 LM을 사용해서 EBR 구현을 보일 것임

            RAG(필요 데이터를 NN로 찾아서 프롬프트에 같이 넣어주는것)는 아니지만, RAG와 비슷하게 chunk를 임베딩해서 검색에 활용

            Stratified sampling에 대해서도 언급해야하고

            Retrieve 단계에선 Recall이 중요하다는것을 precision을 보이면서 해결할것
            검색은 retreive - rerank 단계로 구성되어있음

            "delimiter": [" ", "\n"], # newline, space 차이가 없음
            "grouping": [None, ["idx", "title", "section"], ["idx", "title"]],  idx, title, section이 가장 좋음

            sections: ['인트로' '강의소개' '수강효과' '수강특징' '수강대상' '수강 후기' '질의 응답' '강사소개']
                    {"강사소개": 0.1, "title": 1,2, "강의소개": 1,0.5 "인트로": 1,0.5},


            "st_model": ["jhgan/ko-sroberta-multitask",
                         "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                         "intfloat/multilingual-e5-large",
                         "jhgan/ko-sbert-sts"],
        """
        self.params = {
            "st_model": ["intfloat/multilingual-e5-large"],
            "dataset": {
                "delimiter": [" "],
                "grouping": [["idx", "title", "section"]],
                "section_weight": [
                    {"강사소개": 0.1, "title": 1, "강의소개": 1, "인트로": 1},
                    {"강사소개": 0.1, "title": 2, "강의소개": 1, "인트로": 1},
                    {"강사소개": 0.1, "title": 1, "강의소개": 2, "인트로": 1},
                    {"강사소개": 0.1, "title": 2, "강의소개": 2, "인트로": 1},
                    {"강사소개": 0.1, "title": 1, "강의소개": 1, "인트로": 2},
                    {"강사소개": 0.1, "title": 2, "강의소개": 1, "인트로": 2},
                    {"강사소개": 0.1, "title": 1, "강의소개": 2, "인트로": 2},
                    {"강사소개": 0.1, "title": 2, "강의소개": 2, "인트로": 2},
                ],
                "retrieval_candidate_times": [15, 30, 50]
            }
        }
        self.dataset_path = args.dataset
        self.cases = load_testcases(args.cases_path)
        self.index_root_path = args.index_root_path
        self.index_type = args.index_type
        self.skip_index = args.skip_index

    def vanilla_eval(self, model_name, dataset_param, skip_index=False):

        model = SentenceBert(model_name=model_name)
        dataset = LSDataset(self.dataset_path, dataset_param)

        inference = LSVanilla(model, dataset, 16)
        iter_name = f"{model_name}_{dataset_param}"
        index_fname = f"{iter_name}.pth"
        index_fpath = os.path.join(self.index_root_path, index_fname)
        if skip_index and os.path.isfile(index_fpath):
            print("SKIP to index: " + index_fpath)
        else:
            inference.indexing(index_fpath)

        iter_result_name = f"{iter_name}_vanilla_result.csv"
        iter_result_path = os.path.join(self.index_root_path, iter_result_name)
        if skip_index and os.path.isfile(iter_result_path):
            print("SKIP to evaluate: " + iter_result_name)
            df = pd.read_csv(iter_result_path)
            return df["avg_score"][0]
        else:
            evaluation = LSEvaluation(self.cases, model, dataset, dataset_param["retrieval_candidate_times"])
            score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst = evaluation.vanilla(index_fpath)
            avg_score = 1.0 * sum(score_lst) / len(score_lst)
            evaluation.cases['recall'] = score_lst
            evaluation.cases['retrieved_docs'] = retrieved_docs_lst
            evaluation.cases['expected_details'] = expected_lec_detail_lst
            evaluation.cases['result_details'] = search_result_detail_lst
            evaluation.cases['avg_score'] = avg_score
            evaluation.cases.to_csv(iter_result_path)
            return avg_score

    def faiss_eval(self, model_name, dataset_param, skip_index=False):

        model = SentenceBert(model_name=model_name)
        dataset = LSDataset(self.dataset_path, dataset_param)

        inference = LSFaiss(model, dataset, batch_size=16)
        iter_name = f"{model_name}_{dataset_param}"
        index_fname = f"{iter_name}.index"
        index_fpath = os.path.join(self.index_root_path, index_fname)
        if skip_index is False:
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
        for model in self.params["st_model"]:
            print("MODEL: ", model)
            keys, values = zip(*dataset_params.items())
            for dataset_param in [dict(zip(keys, v)) for v in itertools.product(*values)]:
                if self.index_type == "faiss":
                    score = self.faiss_eval(model, dataset_param, skip_index=self.skip_index)
                else:
                    score = self.vanilla_eval(model, dataset_param, skip_index=self.skip_index)
                result_lst.append([str(score), f"{model}_{dataset_param}"])

            with open(os.path.join(self.index_root_path, model + "_final_result.csv"), "w") as f:
                for result in result_lst:
                    f.write("\t".join(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--index_root_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--index_type", type=str)
    parser.add_argument("--skip_index", type=bool, default=False)
    args = parser.parse_args()
    gs = GridSearch(args)
    gs.explore()
