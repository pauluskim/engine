import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from onnxruntime import InferenceSession
from transformers.convert_graph_to_onnx import convert

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer

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
        path_obj = Path(path)
        if(path_obj.parent.exists()):
            shutil.rmtree(path_obj.parent)
        path_obj.parent.mkdir()
        convert(framework="pt", model=model_name, output=Path(path), opset=11)

    # def load_onnx(self, onnx_path, tokenizer_path):
    #     sample_query = "러닝스푼즈"
    #
    #     sess = InferenceSession(onnx_path)
    #
    #     tokenizer =
    #     model_inputs =
    #     onnx_output =
    #
    #     onnx_embeddings = # average pooling
    #     onnx_embeddings = # normalized
    #
    #     # Actual part
    #     actual_output = self.model.encode(sample_query)
    #     actual_tensor = torch.tensor(actual_output).view(1,-1)
    #
    #     assert (abs(1.0 - torch.cosine_similarity(onnx_embeddings, actual_tensor).tolist()[0]) < 1e-3)
    #
    # def average_pool(self, onnx_output, attention_mask):
    #     pass


if __name__ == "__main__":
    model = SentenceBert(model_name="intfloat/multilingual-e5-large")
    # model.export("/Users/jack/engine/learning_spoons/resource/ST")
    # model.export_onnx("intfloat/multilingual-e5-large", "/Users/jack/engine/learning_spoons/resource/export_new/model.onnx")
    model.load_onnx("/Users/jack/engine/learning_spoons/resource/export_new/model.onnx",
                        "/Users/jack/engine/learning_spoons/resource/export/tokenizer")
