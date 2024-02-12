
import pandas as pd
from torch.utils.data import Dataset

from data.utils import timeit


class LSDataset(Dataset):
    def __init__(self, fpath, params, tokenizer=None):
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

        self.tokenizer = tokenizer

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
        if fields is None:
            self.refined_df = self.df
        else:
            self.refined_df = self.df.groupby(fields, as_index=False).agg({"text": " ".join, "section_weight": "first"})
        self.refined_columns2idx = {col_name: idx for idx, col_name in enumerate(list(self.refined_df.columns))}

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
        lec_id = row[self.refined_columns2idx["idx"]]
        lec_title = row[self.refined_columns2idx["title"]]
        text = row[self.refined_columns2idx["text"]]
        section = row[self.refined_columns2idx["section"]] if "section" in self.refined_columns2idx else "NA"
        section_weight = row[self.refined_columns2idx["section_weight"]]
        return [lec_id, lec_title, text, section, section_weight]

    def get_by_lec_id(self, lec_id):
        return self.refined_df[self.refined_df["idx"] == lec_id].values.tolist()

