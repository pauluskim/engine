from sentence_transformers import SentenceTransformer


class SentenceBert:
    def __init__(self):
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def infer(self, query):
        return self.model.encode(query, convert_to_tensor=True)


if __name__ == "__main__":
    model = SentenceBert()
    print(model.infer("뭐 먹을래?"))

