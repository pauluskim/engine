import pickle

import hnswlib
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.rating_dataset import RatingDataset
from model.sentence_bert import SentenceBert


class Inference():
    def __init__(self, dataset, batch_size):
        self.model = SentenceBert()
        self.data_loader = DataLoader(dataset, batch_size=batch_size)

        self.prepare_index()

    def prepare_index(self):
        test_vector = self.model.infer("test")
        vector_dim = test_vector.size()[0]
        self.p = hnswlib.Index(space="cosine", dim=vector_dim)
        self.p.init_index(max_elements=len(self.data_loader), ef_construction=200, M=16)

    def index(self, output_path):
        for doc_ids, contexts in tqdm(self.data_loader, desc="Index vectors"):
            vectors = self.model.infer(contexts)
            self.p.add_items(vectors, doc_ids)
            break
        self.p.set_ef(50)
        self.save(output_path)

    def save(self, output_path):
        with open(output_path, "wb") as f:
            pickle.dump(self.p, f)


if __name__ == "__main__":
    rating_dataset = RatingDataset()
    inference = Inference(rating_dataset, batch_size=100)
    index_fpath = "/Users/jack/engine/resource/index/rating.pickle"
    inference.index(index_fpath)
