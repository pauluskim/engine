import argparse
import ast
from collections import Counter, defaultdict

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
        """
        lecture의 최종 스코어를 계산

        lecture_score = sum(weighted_doc_score)
        weighted_doc_score = doc_score * doc_weight(section_weight)

        return candidates, search_context

        # Caveat
        section 수에 따른 ranking에 왜곡이 있을 수 있음
        A라는 강의는 0.7의 section이 5개
        B라는 강의는 0.8의 section이 3개 라고 하면
        현재 논리에서는 A가 더 높은 점수를 차지함. 따라서 section의 제한을 거는게 필요
        retrieve 단계에서는 크게 문제될것은 아님. 이건 reranking 단에서 해결해야할 문제
        """
        lec_scores = Counter()
        search_context = defaultdict(list)
        for doc_id, score in zip(doc_ids, scores):
            pass

        return None, None

    def recall(self, expected_lst, actual_lst):
        pass

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