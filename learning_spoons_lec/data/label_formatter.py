import ast
import os

import pandas as pd


def main():
    db_df = load_db().reset_index(drop=True)
    model_result = read_model_result()
    merged_df = pd.merge(model_result, db_df, left_on='docs', right_on='idx')
    merged_df.drop(["idx"], axis=1, inplace=True)
    merged_df["relevance"] = ""
    merged_df = merged_df[['query', 'docs', 'relevance', 'title', '인트로',
                           '강의소개', '강사소개', '수강 후기', '수강대상',
                           '수강특징', '수강효과', '질의 응답']]
    merged_df.to_csv(label_path, sep='\t')


def load_db():
    db_df = pd.read_parquet(db_path)

    title_df = db_df.groupby(["idx", "title"]).first().reset_index()
    title_df["text"] = title_df["title"]
    title_df["section"] = "title"

    db_df = pd.concat([db_df, title_df], ignore_index=True)

    db_df.drop(db_df[db_df['text'].isnull()].index, inplace=True)
    grouped_db_df = db_df.groupby(["idx", "section"], as_index=False).agg({"text": " ".join})
    return grouped_db_df.pivot_table('text', ['idx'], 'section', aggfunc=lambda x: ' '.join(x), fill_value='NA').reset_index()

def read_model_result(topk=15):
    df = pd.read_csv(fpath, delimiter="\t")
    df.drop(["idx", "recall"], axis=1, inplace=True)
    df['docs'] = df['retrieved_docs'].apply(lambda x: ast.literal_eval(x)[:topk])
    df = df.explode('docs', ignore_index=True)
    df.drop(["retrieved_docs"], axis=1, inplace=True)
    return df



if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.realpath(__file__))
    ls_index_path = os.path.join(curr_path, "../resource/ls_index")
    fname = "large_lm_01_1_2_1_30.csv"
    fpath = os.path.join(ls_index_path, fname)
    db_path = os.path.join(curr_path, "../resource/learningspoons_data.parquet")
    label_path = os.path.join(curr_path, "../resource/precision_label.csv")
    main()