import argparse
import os
import pdb
import ast
import pickle
from collections import Counter, defaultdict

import numpy as np
from faiss import read_index
from tqdm import tqdm

from data.ls_dataset import LSDataset
from data.utils import load_args

from model.sentence_bert import SentenceBert

import torch
from torch.nn import functional


class LSEvaluation:
    def __init__(self, cases, model, dataset, retrieval_candidate_times):
        self.cases = cases
        self.model = model
        self.dataset = dataset
        self.retrieval_candidate_times = retrieval_candidate_times
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # def hnsw(self, index_path, output_path):
    #     with open(index_path, "rb") as f:
    #         corpus_embeddings = pickle.load(f)
    #
    #     score_lst = []
    #     retrieved_docs_lst = []
    #     expected_lec_detail_lst = []
    #     search_result_detail_lst = []
    #     for _, row in tqdm(self.cases.iterrows(), desc="Evaluation"):
    #         query = row["query"]
    #         target_docs = ast.literal_eval(row["idx"])
    #
    #         query_vector = self.model.infer(query).cpu()
    #         # expand dim for query vector
    #         corpus_ids, distances = corpus_embeddings.knn_query(query_vector,
    #                                                          k=min(
    #                                                              max(len(target_docs), self.retrieval_candidate_times * 2),
    #                                                              len(self.dataset)
    #                                                          )
    #                                                             )
    #
    #         scores = 1 / (1 +distances[0])
    #         # https://stats.stackexchange.com/a/158285
    #         ranked_lectures, search_context = self.postprocess(corpus_ids[0], scores)
    #         ranked_lecture_idxs = [doc_idx for doc_idx, score in ranked_lectures]
    #         ranked_lecture_idxs = ranked_lecture_idxs[:max(len(target_docs), self.retrieval_candidate_times)]
    #
    #         score = self.recall(target_docs, ranked_lecture_idxs)
    #         score_lst.append(score)
    #         retrieved_docs_lst.append(ranked_lecture_idxs)
    #
    #         expected_lec_details = self.get_search_context_for_target_lec(query_vector, target_docs)
    #         search_result_details = self.get_search_context_for_search_result(ranked_lectures, search_context)
    #         expected_lec_detail_lst.append(expected_lec_details)
    #         search_result_detail_lst.append(search_result_details)
    #
    #     avg_score = 1.0 * sum(score_lst) / len(score_lst)
    #     self.save_as_csv(output_path, score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst,
    #                      avg_score)
    #     return avg_score
    #
    # def faiss(self, index_path, output_path):
    #     index = read_index(index_path)
    #     score_lst = []
    #     retrieved_docs_lst = []
    #     expected_lec_detail_lst = []
    #     search_result_detail_lst = []
    #     for _, row in tqdm(self.cases.iterrows(), desc="Evaluation"):
    #         query = row["query"]
    #         target_docs = ast.literal_eval(row["idx"])
    #
    #         query_vector = self.model.infer(query).cpu()
    #         # expand dim for query vector
    #         scores, corpus_ids = index.search(np.expand_dims(query_vector, axis=0),
    #                                           min(
    #                                               max(len(target_docs), self.retrieval_candidate_times * 2),
    #                                               len(self.dataset))
    #                                           )
    #         ranked_lectures, search_context = self.postprocess(corpus_ids[0], scores[0])
    #         ranked_lecture_idxs = [doc_idx for doc_idx, score in ranked_lectures]
    #         ranked_lecture_idxs = ranked_lecture_idxs[:max(len(target_docs), self.retrieval_candidate_times)]
    #
    #         score = self.recall(target_docs, ranked_lecture_idxs)
    #         score_lst.append(score)
    #         retrieved_docs_lst.append(ranked_lecture_idxs)
    #
    #         expected_lec_details = self.get_search_context_for_target_lec(query_vector, target_docs)
    #         search_result_details = self.get_search_context_for_search_result(ranked_lectures, search_context)
    #         expected_lec_detail_lst.append(expected_lec_details)
    #         search_result_detail_lst.append(search_result_details)
    #
    #     avg_score = 1.0 * sum(score_lst) / len(score_lst)
    #     self.save_as_csv(output_path, score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst,
    #                      avg_score)
    #     return avg_score

    # def vanilla(self, index_path, output_path):
    #     corpus_embeddings = torch.load(index_path, map_location=torch.device(self.device))
    #     score_lst = []
    #     retrieved_docs_lst = []
    #     expected_lec_detail_lst = []
    #     search_result_detail_lst = []
    #     for _, row in tqdm(self.cases.iterrows(), desc="Evaluation"):
    #         query = row["query"]
    #         target_docs = ast.literal_eval(row["idx"])
    #
    #         query_embedding = self.model.infer(query)
    #         query_embedding = functional.normalize(query_embedding, p=2.0, dim=0)
    #
    #         # Already query_embedding and corpus_embeddings are normalized.
    #         cos_scores = torch.inner(query_embedding, corpus_embeddings)
    #
    #         # Find the top k docs from the calculation results.
    #         scores, doc_idxs = torch.topk(cos_scores,
    #                                       k=min(
    #                                           max(len(target_docs), self.retrieval_candidate_times * 2),
    #                                           len(cos_scores)))
    #
    #         candidates, search_context = self.postprocess(doc_idxs.cpu(), scores.cpu())
    #         candidates_idxs = [lec_idx for lec_idx, score in candidates]
    #         candidates_idxs = candidates_idxs[:max(len(target_docs), self.retrieval_candidate_times)]
    #
    #         score = self.recall(target_docs, candidates_idxs)
    #         score_lst.append(score)
    #         retrieved_docs_lst.append(candidates_idxs)
    #
    #         # For debugging
    #         target_lec_details = self.get_search_context_for_target_lec(query_embedding.cpu(), target_docs)
    #         search_result_details = self.get_search_context_for_search_result(candidates, search_context)
    #         expected_lec_detail_lst.append(target_lec_details)
    #         search_result_detail_lst.append(search_result_details)
    #
    #     avg_score = 1.0 * sum(score_lst) / len(score_lst)
    #     self.save_as_csv(output_path, score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst,
    #                      avg_score)
    #     return avg_score

    def main(self, index, output_path):
        score_lst = []
        retrieved_docs_lst = []
        expected_lec_detail_lst = []
        search_result_detail_lst = []
        for _, row in tqdm(self.cases.iterrows(), desc="Evaluation"):
            query = row["query"]
            target_docs = ast.literal_eval(row["idx"])

            query_embedding = self.model.infer(query)
            query_embedding = functional.normalize(query_embedding, p=2.0, dim=0)

            doc_idxs, scores = index.search(query_embedding, min(
                                          max(len(target_docs), self.retrieval_candidate_times * 2),
                                          len(self.dataset)))

            candidates, search_context = self.postprocess(doc_idxs, scores)
            candidates_idxs = [lec_idx for lec_idx, score in candidates]
            candidates_idxs = candidates_idxs[:max(len(target_docs), self.retrieval_candidate_times)]

            score = self.recall(target_docs, candidates_idxs)
            score_lst.append(score)
            retrieved_docs_lst.append(candidates_idxs)

            # For debugging
            target_lec_details = self.get_search_context_for_target_lec(query_embedding.cpu(), target_docs)
            search_result_details = self.get_search_context_for_search_result(candidates, search_context)
            expected_lec_detail_lst.append(target_lec_details)
            search_result_detail_lst.append(search_result_details)

        avg_score = 1.0 * sum(score_lst) / len(score_lst)
        self.save_as_csv(output_path, score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst,
                         avg_score)
        return avg_score

    def postprocess(self, doc_ids, scores):
        lec_scores = Counter()
        search_context = defaultdict(list)
        for doc_id, score in zip(doc_ids, scores):
            lec_id, lec_title, text, section, section_weight = self.dataset[doc_id]
            # 근데 여기서, section 수에 따른 ranking에 왜곡이 있을 수 있음
            # 가령 A라는 강의는 0.7의 section이 5개
            # B라는 강의는 0.8의 section이 3개 라고 하면 현재 논리에서는 A가 더 높은 점수를 차지함. 따라서 section의 제한을 거는게 필요할텐데
            # retrieve 단계에서는 크게 문제될것은 아님. 이건 reranking 단에서 해결해야할 문제로 보임
            lec_scores[lec_id] += score * section_weight
            search_context[lec_id].append([lec_title, text, section, score * section_weight])

        return sorted(lec_scores.items(), key=lambda item: -item[1]), search_context

    def get_search_context_for_target_lec(self, query_vector, target_docs):
        lec_info_dict = dict()
        for lec_id in target_docs:
            docs = self.dataset.get_by_lec_id(lec_id)

            texts = []
            section_weights = []
            for doc in docs:
                text_idx = self.dataset.refined_columns2idx["text"]

                texts.append(doc[text_idx])
                section_weights.append(doc[-1])

            vectors = self.model.infer(texts).cpu()
            normed_vectors = functional.normalize(vectors, p=2.0, dim=1)
            scores = torch.inner(normed_vectors, query_vector)
            weighted_scores = scores * torch.tensor(section_weights)

            for i in range(len(docs)):
                doc = docs[i]
                weighted_score = weighted_scores[i]
                doc.append(weighted_score)
                docs[i] = doc

            lec_info_dict[lec_id] = docs
        return lec_info_dict

    def get_search_context_for_search_result(self, ranked_lectures, search_context):
        lec_info_dict = dict()
        for lec_id, score in ranked_lectures:
            lec_info = []
            for lec_title, text, section, weighted_score in search_context[lec_id]:
                lec_info.append([lec_title, section, text, weighted_score])
            lec_info_dict[lec_id] = lec_info
        return lec_info_dict

    def recall(self, expected_lst, actual_lst):
        actual_set = set(actual_lst)
        recall_cnt = 0
        for expected in expected_lst:
            if expected in actual_set:
                recall_cnt += 1

        return 1.0 * recall_cnt / len(expected_lst)

    def save_as_csv(self, path, score_lst, retrieved_docs_lst, expected_lec_detail_lst, search_result_detail_lst,
                    avg_score):
        self.cases['recall'] = score_lst
        self.cases['retrieved_docs'] = retrieved_docs_lst
        self.cases['expected_details'] = expected_lec_detail_lst
        self.cases['result_details'] = search_result_detail_lst
        self.cases['avg_score'] = avg_score
        self.cases.to_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=str)
    parser.add_argument("--faiss_index", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    cases = load_args(args.cases_path)
    model = SentenceBert()
    dataset = LSDataset(args.dataset)

    evaluation = LSEvaluation(cases, model, dataset)
    evaluation.faiss(args.faiss_index)
