import argparse
import sys

import numpy as np
import tritonclient.http as httpclient

def infer(triton_client, model_name, query):
    inputs = []
    input_tensor = np.array([[query.encode('utf-8')]])
    input = httpclient.InferInput("query", input_tensor.shape, "BYTES")
    input.set_data_from_numpy(input_tensor)
    inputs.append(input)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput("reviews", binary_data=False))
    outputs.append(httpclient.InferRequestedOutput("dists", binary_data=False))

    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    reviews = results.as_numpy("reviews").tolist().split("*&*")
    dists = results.as_numpy("dists").tolist()

    return reviews, dists

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

    queries = ["재밌는 영화", "여주인공이 예뻣던 영화", "킬링 타임용 영화", "답답해서 암걸릴 것 같은 영화"]

    for query in queries:
        reviews, dists = infer(triton_client, "ebr", query)
        print("Query: {}".format(query))
        for review, dist in zip(reviews, dists):
            print("\t{}: {}".format(review, dist))