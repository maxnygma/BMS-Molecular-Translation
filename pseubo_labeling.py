import pandas as pd

import torch

import shutil
import re
from os.path import isfile
from tqdm.auto import tqdm
tqdm.pandas()

from config import config
from data import Tokenizer


def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')


def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')


def get_test_file_path(image_id):
    return "data/test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )


def get_pseudo_labeled_images(config):
    df_is_valid = pd.read_csv(config.paths.path_to_pseudo_labels_csv)
    df = pd.read_pickle(config.paths.path_to_csv)

    tokenizer = torch.load(config.paths.path_to_tokenizer_weights)

    df_is_valid['image_path'] = df_is_valid['image_id'].apply(get_test_file_path)
    df_is_valid['InChI_1'] = df_is_valid['InChI'].progress_apply(lambda x: x.split('/')[1])
    df_is_valid['InChI_text'] = df_is_valid['InChI_1'].progress_apply(split_form) + ' ' + \
                            df_is_valid['InChI'].apply(lambda x: '/'.join(x.split('/')[2:])).progress_apply(split_form2).values

    lengths = []
    for text in tqdm(df_is_valid['InChI_text'].values, total=len(df_is_valid)):
        try:
            seq = tokenizer.text_to_sequence(text)
        except:
            lengths.append(999)
            continue

        length = len(seq) - 2
        lengths.append(length)

    df_is_valid['InChI_length'] = lengths
    df_is_valid = df_is_valid.drop(df_is_valid[df_is_valid['InChI_length'] == 999].index)

    final_dict = {}
    for i, x in tqdm(enumerate(range(len(df_is_valid)))):
        if df_is_valid['is_valid'].iloc[x] == True:
            row = df_is_valid.iloc[x]

            final_dict[i] = {'image_id': row['image_id'], 'InChI': row['InChI'], 'InChI_1': row['InChI_1'], 'InChI_text': row['InChI_text'], 'InChI_length': row['InChI_length'], 'image_path': row['image_path']}
    
    pseudo_labels_df = pd.DataFrame.from_dict(final_dict, orient='index')
    print(pseudo_labels_df)
    df = pd.concat([df, pseudo_labels_df], ignore_index=False)

    return df

df1 = pd.read_pickle(config.paths.path_to_csv)
print(len(df1))

df = get_pseudo_labeled_images(config)
df.to_pickle('train_pseudo_labels.pkl')
print(len(df))

