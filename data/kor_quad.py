import json
import os
from os.path import isfile, join


class KorQuad:
    def __init__(self, data_path: str):
        if data_path.endswith(".json"):
            self.f_lst = [data_path]
        else:
            assert os.path.isdir(data_path)
            self._lst = [f for f in os.listdir(data_path) if f.endswith(".json")]

    def get_a_sample(self):
        with open(self.f_lst[0], "r", encoding="utf-8") as f:
            json_obj = json.load(f)
        print(json.dumps(json_obj['data'][0], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    data = KorQuad(data_path="/Users/jack/engine/resource/quad/v1.0/KorQuAD_v1.0_train.json")
    data.get_a_sample()