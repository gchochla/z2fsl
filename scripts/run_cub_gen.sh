#!/bin/bash

while getopts m:d:c: flag
do
    case "${flag}" in
        m) model_dir=${OPTARG};;
        d) dataset_dir=${OPTARG};;
        c) device=${OPTARG};;
    esac
done

python z2fsl/z2fsl_vaegan.py -dt -dd "$dataset_dir" -gen True -fsld "$model_dir" \
-tfsl False -fn -glr 1e-4 -ghl 4096-8192 -clsfhl '' -s 5 -w 25 -vs 1800 -tvs 5 \
-z2f 10 -wgan 100 -eps 6500 -rs True -dvc $device
