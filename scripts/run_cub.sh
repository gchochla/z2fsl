#!/bin/bash

while getopts m:d:c: flag
do
    case "${flag}" in
        m) model_dir=${OPTARG};;
        d) dataset_dir=${OPTARG};;
        c) device=${OPTARG};;
    esac
done

python z2fsl/z2fsl_vaegan.py -dt -dd "$dataset_dir" -fsld "$model_dir" \
-tfsl True -fn -glr 1e-4 -flr 5e-5 -ghl 4096-8192 -clsfhl '' -s 5 -w 25 \
-z2f 100 -wgan 100 -eps 8000 -vs 1800 -dvc $device
