import json
import os
import pdb
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import triton_python_backend_utils as pb_utils
from faiss import read_index
from onnxruntime import InferenceSession


class LSDataset:
    def __init__(self, fpath, params):
        """
            params:
                delimiter: " ", "\n"
                grouping: ["idx", "title"], ["idx", "title", "section"], non_grouping,
                section_weight: {"강사소개": 0.1}
        """

        self.df = pd.read_parquet(fpath)
        self.df = self.add_title_as_text()

        self.df.drop(self.df[self.df['text'].isnull()].index, inplace=True)
        self.df['text'] = (self.df['text'].str
                           .replace('$%^', params["delimiter"], regex=False)
                           .replace('!$%^', params["delimiter"], regex=False))
        self.set_section_weight(params["section_weight"])
        self.set_refined_df_by_grouping(params["grouping"])

    def add_title_as_text(self):
        df_by_lec = self.df.groupby(["idx", "title"]).first().reset_index()
        df_by_lec["text"] = df_by_lec["title"]
        df_by_lec["section"] = "title"
        return pd.concat([self.df, df_by_lec], ignore_index=True)

    def set_section_weight(self, section_weight_map):
        self.df["section_weight"] = 1.0
        for section, weight in section_weight_map.items():
            self.df.loc[self.df["section"] == section, 'section_weight'] = weight

    def set_refined_df_by_grouping(self, fields):
        self.refined_df = self.df.groupby(fields, as_index=False).agg({"text": " ".join, "section_weight": "first"})
        self.refined_columns2idx = {col_name: idx for idx, col_name in enumerate(list(self.refined_df.columns))}

    def __getitem__(self, index):
        row = self.refined_df.iloc[[index]].values[0].tolist()
        lec_id = row[self.refined_columns2idx["idx"]]
        lec_title = row[self.refined_columns2idx["title"]]
        text = row[self.refined_columns2idx["text"]]
        section = row[self.refined_columns2idx["section"]] if "section" in self.refined_columns2idx else "NA"
        section_weight = row[self.refined_columns2idx["section_weight"]]
        return [lec_id, lec_title, text, section, section_weight]

class TritonPythonModel:
    def initialize(self, args):
        print('Initialized...')
        # Load config
        model_config = json.loads(args["model_config"])
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        title_output_config = pb_utils.get_output_config_by_name(model_config, "lec_titles")
        self.title_dtype = pb_utils.triton_string_to_numpy(title_output_config["data_type"])
        score_output_config = pb_utils.get_output_config_by_name(model_config, "scores")
        self.score_dtype = pb_utils.triton_string_to_numpy(score_output_config["data_type"])

        # Load Onnxruntime session
        onnx_path = os.path.join(model_dir, "onnx/model.onnx")
        self.sess = InferenceSession(onnx_path)

        # Load Faiss index
        index_path = os.path.join(model_dir, model_config["parameters"]["index"]["string_value"])
        self.index = read_index(index_path)

        # Load dataset
        dataset_path = os.path.join(model_dir, model_config["parameters"]["dataset"]["string_value"])
        dataset_param = {
            "delimiter": " ",
            "grouping": ["idx", "title", "section"],
            "section_weight": {"강사소개": 0.1}
        }
        self.dataset = LSDataset(dataset_path, dataset_param)


    def execute(self, requests):
        responses = []
        for request in requests:
            # Need to get user query from the request
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            onnx_output = self.sess.run(None, model_inputs)
            onnx_embeddings = self.average_pool(onnx_output, model_inputs['attention_mask'])
            onnx_embeddings = F.normalize(onnx_embeddings, p=2, dim=1)

            scores, doc_idxs = self.index.search(onnx_embeddings, 60)

            candidates = self.postprocess(doc_idxs[0], scores[0])
            ranked_scores = []
            ranked_lecs = []
            for lec_title, score in candidates:
                ranked_lecs.append(lec_title)
                ranked_scores.append(score)

            lec_tensor = pb_utils.Tensor("lec_titles",
                                            np.array("*&*".join(ranked_lecs), dtype=self.title_dtype))
            score_tensor = pb_utils.Tensor("scores", np.array(ranked_scores, dtype=self.score_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[lec_tensor, score_tensor])
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def average_pool(self, onnx_output, attention_mask):
        last_hidden_states = torch.from_numpy(onnx_output[0])
        attention_mask = torch.from_numpy(attention_mask)
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def postprocess(self, doc_ids, scores):
        lec_scores = Counter()
        for doc_id, score in zip(doc_ids, scores):
            lec_id, lec_title, text, section, section_weight = self.dataset[doc_id]
            lec_scores[lec_title] += score * section_weight

        return sorted(lec_scores.items(), key=lambda item: -item[1])

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
