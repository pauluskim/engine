from unittest import TestCase

from sentence_transformers import SentenceTransformer


class TestSentenceBert(TestCase):
    def setUp(self) -> None:
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def test_infer(self):
        sentences = ['This framework generates embeddings for each input sentence',
                     'Sentences are passed as a list of string.',
                     'The quick brown fox jumps over the lazy dog.']
        embeddings = self.model.encode(sentences)
        # Print the embeddings
        for sentence, embedding in zip(sentences, embeddings):
            print("Sentence:", sentence)
            print("Embedding:", embedding)
