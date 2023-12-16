import argparse
from collections import Counter, defaultdict

import numpy as np
from faiss import read_index
from sklearn.metrics import dcg_score
from torch.utils.data import DataLoader

from data.ls_dataset import LSDataset
from data.utils import load_args
from model.sentence_bert import SentenceBert


class LSEvaluation:
    def __init__(self, cases, model, dataset):
        self.cases = cases
        self.model = model
        self.dataset = dataset

    def faiss(self, index):
        index = read_index(index)

        for case in self.cases:
            query = case["query"]
            retrieved_docs = case["lec_ids"]

            query_vector = self.model.infer(query)
            # expand dim for query vector
            query_vectory = np.expand_dims(query_vector, axis=0)
            scores, corpus_ids = index.search(query_vectory, len(retrieved_docs))

            ranked_lectures, search_context = self.postprocess(corpus_ids[0], scores[0])

            self.print_search_result(query, ranked_lectures, search_context)


    def postprocess(self, doc_ids, scores):
        lec_scores = Counter()
        search_context = defaultdict(list)
        for doc_id, score in zip(doc_ids, scores):
            text_id, lec_id, lec_title, text, section_weight = self.dataset[doc_id]
            lec_scores[lec_id] += score * section_weight
            search_context[lec_id].append([lec_title, text, score * section_weight])

        return sorted(lec_scores.items(), key=lambda item: -item[1]), search_context

    def print_search_result(self, query, ranked_lectures, search_context):
        print("-----------------------------------------------------")
        print(f"Query: {query}")
        for lec_id, score in ranked_lectures:
            title = search_context[lec_id][0][0]
            text = ", ".join([f"{context[1]}:{context[2]}" for context in search_context[lec_id]])
            print(f"\ttitle: {title}")
            print(f"\tscore: {score}")
            print(f"\tRAG: {text}")
            print()

    def dcg(self, expected, corpus_ids, distances):

        dcg_score(expected, [])

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