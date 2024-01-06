import time

import pandas as pd
from torch.utils.data import Dataset

from data.utils import timeit
import pdb


class LSDataset(Dataset):
    def __init__(self, fpath, params, tokenizer=None):
        """
            Need to compare the precision if the length is over than 128 which is the max_seq_len of the model.

            params:
                delimiter: " ", "\n"
                grouping: ["idx", "title"], ["idx", "title", "section"], non_grouping,
                section_weight: {"강사소개": 0.1}
        """

        self.df = pd.read_parquet(fpath)
        self.df.drop(self.df[self.df['text'].isnull()].index, inplace=True)
        self.df['text'] = self.df['text'].str.replace('!$%^', params["delimiter"], regex=False)
        self.section_weight = params["section_weight"]

        self.set_refined_df_by_grouping(params["grouping"])

        self.tokenizer = tokenizer

    def set_refined_df_by_grouping(self, fields):
        if fields is None:
            self.refined_df = self.df
        else:
            self.refined_df = self.df.groupby(fields, as_index=False).agg({"text": " ".join})

    @timeit
    def get_max_seq_len_series(self):
        seq_len_series = self.refined_df["text"].apply(lambda col: len(self.tokenizer(col)[0]))
        return max(seq_len_series)

    @timeit
    def get_max_seq_len_df(self):
        seq_len_df = self.refined_df.apply(lambda row: len(self.tokenizer(row["text"])[0]), axis=1)
        return max(seq_len_df)


    def __len__(self):
        return len(self.refined_df)

    def __getitem__(self, index):
        row = self.refined_df.iloc[[index]].values[0].tolist()
        lec_id = row[0]
        lec_title = row[1]
        section = row[2]
        text_id = index
        text = row[-1]
        return [text_id, lec_id, lec_title, text, self.section_weight.get(section, 1)]
