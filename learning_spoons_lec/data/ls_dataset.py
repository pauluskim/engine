
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

        # 1. text field null check
        # 2. text delimiter sets
        # 3. section weight 설정

        # 4. Text chunk GroupBy
        self.set_refined_df_by_grouping(params["grouping"])

        # Model별로 seq limit이 존재함. 이론상 seq limit이 없지만, limit을 넘어가는 문장에 대하 performanc e가 떨어지기도 함
        # seq len를 구하기 위한 tokenizer
        self.tokenizer = tokenizer

    def add_title_as_text(self):
        """
        "title" field에 있는 title을 기존 data format에 맞춰서 reformat
        """
        pass


    def set_refined_df_by_grouping(self, fields):
        # 1. GroupBy
        # 2. 추후에 column(field) index가 필요. 하지만 Grouping을 하다보면, column의 index들은 변경됨.
        #    즉, refined_df의 column index를 저장하는 variable이 필요
        pass

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
        pass

    def get_by_lec_id(self, lec_id):
        rows = self.refined_df[self.refined_df["idx"] == lec_id].values.tolist()
        return [row + [self.section_weight.get(row[2], 1)] for row in rows]

