# Copyright 2016 TensorLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

# prepare.py
# Implements data preparation for the census model.

import json
import numpy as np
import os
import pandas as pd
import tensorfx as tfx
import urllib


NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
         'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss',
         'hours_per_week', 'native_country', 'income_bracket']

TYPES = [np.int32, np.str, np.int32, np.str, np.int32, np.str,
         np.str, np.str, np.str, np.str, np.int32, np.int32,
         np.int32, np.str, np.str]
 

def create_dataframe(url, raw_path, data_path, lines_to_skip=0):
  urllib.urlretrieve(url, raw_path)

  # Load data, while also stripping leading spaces and converting to ? to missing values
  df = pd.read_csv(raw_path, names=NAMES,
                   skiprows=lines_to_skip,  skipinitialspace=True,
                   na_values=['?'])

  for name, dtype in zip(NAMES, TYPES):
    if dtype == np.str:
      df[name] = df[name].astype('category')

  # Drop useless/redundant columns
  df.drop('fnlwgt', 1, inplace=True)
  df.drop('education', 1, inplace=True)

  # Order columns so that the target label is the first column
  cols = df.columns.tolist()
  cols = cols[-1:] + cols[0:-1]
  df = df[cols]

  df.to_csv(data_path, header=False, index=False)
  return df


def create_schema(df, path):
  schema_fields = []
  for name, dtype in zip(df.columns, df.dtypes):
    if type(dtype) == pd.types.dtypes.CategoricalDtype:
      field_type = tfx.data.SchemaFieldType.discrete
    elif dtype == np.int64 or dtype == np.int32 or dtype == np.float64 or dtype == np.float32:
      field_type = tfx.data.SchemaFieldType.numeric

    field = tfx.data.SchemaField(name, field_type)
    schema_fields.append(field)

  schema = tfx.data.Schema(schema_fields)
  with open(path, 'w') as f:
    f.write(schema.format())


def create_metadata(df, path):
  metadata = {}
  for name, dtype in zip(df.columns, df.dtypes):
    md = {}
    if type(dtype) == pd.types.dtypes.CategoricalDtype:
      entries = list(df[name].unique())
      if np.nan in entries:
        entries.remove(np.nan)
      md['entries'] = sorted(entries)
    elif dtype in (np.int32, np.int64, np.float32, np.float64):
      for stat, stat_value in df[name].describe().iteritems():
        if stat == 'min':
          md['min'] = stat_value
        if stat == 'max':
          md['max'] = stat_value

    metadata[name] = md

  with open(path, 'w') as f:
    f.write(json.dumps(metadata, separators=(',',':')))


data_path = '/tmp/tensorfx/census/data'
if not os.path.isdir(data_path):
  os.makedirs(data_path)


train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
train_data_raw = os.path.join(data_path, 'train.raw.csv')
train_data_path = os.path.join(data_path, 'train.csv')

eval_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
eval_data_raw = os.path.join(data_path, 'eval.raw.csv')
eval_data_path = os.path.join(data_path, 'eval.csv')

schema_path = os.path.join(data_path, 'schema.yaml')
metadata_path = os.path.join(data_path, 'metadata.json')


df_train = create_dataframe(train_data_url, train_data_raw, train_data_path)
df_eval = create_dataframe(eval_data_url, eval_data_raw, eval_data_path,
                           lines_to_skip=1)

create_schema(df_train, schema_path)
create_metadata(df_train, metadata_path)

