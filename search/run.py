import pickle

import pandas as pd

from data.rating_dataset import RatingDataset
from model.sentence_bert import SentenceBert


class Run:
    def __init__(self, index_path, dataset):
        self.load_index(index_path)
        self.encoder = SentenceBert()
        self.df = dataset.df

    def load_index(self, index_path):
        with open(index_path, "rb") as f:
            self.p = pickle.load(f)

        ### Index parameters are exposed as class properties:
        print(f"Parameters passed to constructor:  space={self.p.space}, dim={self.p.dim}")
        print(f"Index construction: M={self.p.M}, ef_construction={self.p.ef_construction}")
        print(f"Index size is {self.p.element_count} and index capacity is {self.p.max_elements}")
        print(f"Search speed/quality trade-off parameter: ef={self.p.ef}")

    def search(self, query):
        query_vector = self.encoder.infer(query)
        doc_ids, _ =  self.p.knn_query(query_vector, k=10)
        for doc_id in doc_ids[0]:
            print(self.df.loc[self.df['id'] == doc_id]["document"])

if __name__ == "__main__":
    index_path = "/Users/jack/engine/resource/index/rating.pickle"
    process = Run(index_path, RatingDataset())
    process.search("추천할만 영화")