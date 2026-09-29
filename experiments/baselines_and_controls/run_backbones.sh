#!/bin/bash
cd $HOME/ow_vis
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$HOME/ow_vis/.deps
for bb in small bplus large; do
  echo "===== BACKBONE $bb ====="
  $HOME/fattah-venv/bin/python experiments/eval_backbone_val30.py $bb
done
echo ALL_BACKBONES_DONE
