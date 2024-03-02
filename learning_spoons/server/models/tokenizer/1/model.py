import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
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
        model_config = json.loads(args["model_config"])
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        id_output_config = pb_utils.get_output_config_by_name(model_config, "input_ids")
        self.id_dtype = pb_utils.triton_string_to_numpy(id_output_config["data_type"])

        mask_output_config = pb_utils.get_output_config_by_name(model_config, "attention_mask")
        self.mask_dtype = pb_utils.triton_string_to_numpy(mask_output_config["data_type"])


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
            # Need to get user query from the request
            query = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()[0][0].decode('utf-8')
            output_map = self.tokenizer(query)
            ids = output_map["input_ids"]
            mask = output_map["attention_mask"]


            ids_tensor = pb_utils.Tensor("input_ids", np.array([ids], dtype=self.id_dtype))
            mask_tensor = pb_utils.Tensor("attention_mask", np.array([mask], dtype=self.mask_dtype))

            inference_resp = pb_utils.InferenceResponse(output_tensors=[ids_tensor, mask_tensor])
            responses.append(inference_resp)
        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
