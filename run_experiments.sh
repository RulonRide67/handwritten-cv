#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
python train_svm.py
for s in 42 43 44; do python train.py --name resnet18_real_s$s --synthetic none --seed $s; done
python filter_synthetic.py --model resnet18_real_s42
for s in 42 43 44; do
  python train.py --name resnet18_synth_all_s$s --synthetic all --seed $s
  python train.py --name resnet18_synth_filtered_s$s --synthetic filtered --seed $s
done
cp ../models/resnet18_real_s42.pt ../models/resnet18_real.pt
cp ../models/resnet18_synth_filtered_s42.pt ../models/resnet18_synthetic.pt
python make_figures.py
