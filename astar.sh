#!/bin/bash

model_dir=$1
data_dir=$2
sentences_file=$3
abstract_tags=$4

python astar.py --input_file ./$sentences_file --model_dir ./$model_dir --data_dir ./$data_dir --time_out 600 --abstract_tags $abstract_tags --tag_dict_threshold 5 --seed_tag_dict_threshold 3
