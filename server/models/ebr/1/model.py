import json
import os
import pdb

import numpy as np
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer
from faiss import read_index


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    top_k = 5
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initialized...')
        self.args = args
        self.model_config = json.loads(args["model_config"])
        self.load_model()
        self.load_reviews()
        self.load_index()
        self.set_output_dtype()

    def load_model(self):
        print("load EBR model")
        model_dir = os.path.join(self.args['model_repository'], self.args['model_version'])
        model_name = self.model_config["parameters"]["STMODEL"]["string_value"]
        self.model = SentenceTransformer(os.path.join(model_dir, model_name))

    def load_reviews(self):
        print("load review")
        model_dir = os.path.join(self.args['model_repository'], self.args['model_version'])
        idx2review_fname = self.model_config["parameters"]["IDX2REVIEW"]["string_value"]
        review_path = os.path.join(model_dir, idx2review_fname)

        self.idx2review = []
        with open(review_path, "r", encoding='utf-8') as f:
            f.readline()
            for line in f:
                review = line.strip().split("\t")[1]
                self.idx2review.append(review)

    def load_index(self):
        print("load index")
        model_dir = os.path.join(self.args['model_repository'], self.args['model_version'])
        index_fname = self.model_config["parameters"]["INDEX"]["string_value"]
        index_path = os.path.join(model_dir, index_fname)

        self.index = read_index(index_path)

    def set_output_dtype(self):
        dist_config = pb_utils.get_output_config_by_name(self.model_config, "dists")
        self.dist_dtype = pb_utils.triton_string_to_numpy(dist_config["data_type"])

        review_config = pb_utils.get_output_config_by_name(self.model_config, "reviews")
        self.review_dtype = pb_utils.triton_string_to_numpy(review_config["data_type"])


    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them.
        # Reusing the same pb_utils.InferenceResponse object for multiple
        # requests may result in segmentation faults. You should avoid storing
        # any of the input Tensors in the class attributes as they will be
        # overridden in subsequent inference requests. You can make a copy of
        # the underlying NumPy array and store it if it is required.
        for request in requests:
            query = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()[0][0].decode("utf-8")
            query_vec = self.model.encode(query, convert_to_tensor=True)
            query_vec = np.expand_dims(query_vec, axis=0)
            distances, review_ids = self.index.search(query_vec, self.top_k)

            reviews = []
            for review_id in review_ids[0]:
                reviews.append(self.idx2review[review_id])

            dist_tensor = pb_utils.Tensor("dists", np.array(distances[0], dtype=self.dist_dtype))
            review_tensor = pb_utils.Tensor("reviews",
                                            np.array("*&*".join(reviews), dtype=self.review_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[review_tensor, dist_tensor])
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
