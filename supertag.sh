#!/bin/bash

model_dir=$1
test_file=$2

./scripts/run.sh edin.supertagger.MainApply --model_dirs ./$model_dir --input_file_words ./$test_file --output_file ./$model_dir/single_best --output_file_best_k ./$model_dir/best_40 --top_K 40
