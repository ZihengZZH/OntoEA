#!/usr/bin/env bash

set -e

method=ontoea

for dataset in EN_FR_15K_V1 EN_FR_15K_V2 EN_DE_15K_V1 EN_DE_15K_V2 MED_BBK_9K
do
    log_dir=../../output/logs/${method}/${dataset}
    log_name="${log_dir}/${method}_${dataset}_$(date +%m-%d_%H-%M-%S).log"

    mkdir -p ${log_dir}
    echo "dataset: ${dataset}"
    echo "log dir: ${log_dir}"
    echo "${method} log will be saved to ${log_name}"

    CUDA_VISIBLE_DEVICES="3" python3 main_from_args.py ./args/ontoea_args_15K.json ${dataset} 721_5fold/1/ |tee ${log_name}
done