import argparse
import sys

import numpy as np
import tritonclient.http as httpclient

def generate_output_type(model_name):
    outputs = []
    if model_name == "tokenizer":
        outputs.append(httpclient.InferRequestedOutput("input_ids", binary_data=False))
        outputs.append(httpclient.InferRequestedOutput("attention_mask", binary_data=False))
    elif model_name == "ebr":
        outputs.append(httpclient.InferRequestedOutput("lec_titles", binary_data=False))
        outputs.append(httpclient.InferRequestedOutput("scores", binary_data=False))

    return outputs

def postprocess(model_name, output):
    result = []
    if model_name == "tokenizer":
        input_ids = output.as_numpy("input_ids").tolist()
        attendtion_mask = output.as_numpy("attention_mask").tolist()
        print([input_ids, attendtion_mask])
    elif model_name == "ebr":
        titles = output.as_numpy("lec_titles").tolist().split("*&*")
        scores = output.as_numpy("scores").tolist()
        for title, score in zip(titles, scores):
            print("\t{}: {}".format(title, score))

def infer(triton_client, model_name, query):
    # Generate inputs
    inputs = []
    input_array = np.array([[query.encode('utf-8')]])
    input = httpclient.InferInput("query", input_array.shape, "BYTES")
    input.set_data_from_numpy(input_array)
    inputs.append(input)

    # Generate outputs
    outputs = generate_output_type(model_name)

    # Get response from triton
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    # postprocess from results
    print("Query: ", query)
    postprocess(model_name, results)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )

    FLAGS = parser.parse_args()
    try:
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    queries = ["머신러닝", "부동산", "주식", "데이터분석", "회사 성장시키기", "회사 키우기", "UX 분석", "고객경험 분석"]

    for query in queries:
        infer(triton_client, "ebr", query)