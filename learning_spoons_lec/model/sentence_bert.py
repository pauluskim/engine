import os

import torch
from transformers import PreTrainedTokenizerFast, BertTokenizer
from onnxruntime import SessionOptions, InferenceSession
from sentence_transformers import SentenceTransformer
import pdb

from transformers import AutoTokenizer
from txtai.pipeline import HFOnnx

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
    def __init__(self, model_name="jhgan/ko-sroberta-multitask"):
        self.model = SentenceTransformer(model_name)

    def infer(self, query):
        return self.model.encode(query, convert_to_tensor=True)

    def export(self, path):
        self.model.save(path)

        sample_query = "Learning spoons"
        actual_embedding = self.model.encode(sample_query)
        expected_embedding = SentenceTransformer(path).encode(sample_query)
        assert all([actual == expected for actual, expected in zip(actual_embedding, expected_embedding)])

    def export_onnx(self, model_name, path):
        sample_query = "러닝스푼즈"

        # Onnx part
        tokenizer_path = os.path.join(path, "tokenizer")
        self.model.tokenizer.save_pretrained(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model_inputs = tokenizer(sample_query, return_tensors="np")

        if not os.path.exists(os.path.join(path, "onnx/")):
            os.makedirs(os.path.join(path, "onnx/"))
        onnx_path = os.path.join(path, "onnx/model.onnx")
        onnx = HFOnnx()
        # Converting Model
        onnx_model = onnx(
            model_name,
            "pooling",
            onnx_path,
            quantize=False)
        # ====================================
        # # Implementation
        # ====================================
        options = SessionOptions()
        session = InferenceSession(onnx_model, options)
        onnx_output = session.run(None, dict(model_inputs))
        onnx_tensor = torch.tensor(onnx_output[0][0]).view(1,-1)

        # Actual part
        actual_output = self.model.encode(sample_query)
        actual_tensor = torch.tensor(actual_output).view(1,-1)

        assert (1 == torch.cosine_similarity(onnx_tensor, actual_tensor))


if __name__ == "__main__":
    model = SentenceBert(model_name="intfloat/multilingual-e5-large")
    print(model.infer("learning spoons"))
    # model.export("/Users/jack/engine/learning_spoons_lec/resource/ST")
    model.export_onnx("intfloat/multilingual-e5-large", "/Users/jack/engine/learning_spoons_lec/resource/export")
