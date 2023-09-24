from sentence_transformers import SentenceTransformer

"""
@misc{kr-sbert,
  author = {Park, Suzi and Hyopil Shin},
  title = {KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {{https://github.com/snunlp/KR-SBERT}}
}
"""


class SentenceBert:
    def __init__(self):
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def infer(self, query):
        return self.model.encode(query, convert_to_tensor=True)

    def export(self, path):
        pass

if __name__ == "__main__":
    model = SentenceBert()
    print(model.infer("learning spoons"))
    model.export("/Users/jack/engine/resource/ST")
