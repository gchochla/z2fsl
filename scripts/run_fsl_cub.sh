#!/bin/bash

while getopts m:d:c flag
do
    case "${flag}" in
        m) model_dir=${OPTARG};;
        d) dataset_dir=${OPTARG};;
        c) device=${OPTARG};;
    esac
done

python z2fsl/pretraining/fsl_diagonal.py -dd "$dataset_dir" -mf "$model_dir" \
-fn -gen False -lr 5e-5 -w 25 -s 5 -qp 10 -n 0 -dvc $device \
-spl trainval -eps 12000
