#!/bin/bash
# Waits for the DAVIS and LVOS downloads, unzips them under x3data, builds evaluation roots, writes READY flags.
D=$HOME/ow_vis/x3data; cd $D/dl
while pgrep -u "$USER" -f "DAVIS-2017-trainval-480p.zip" >/dev/null; do sleep 30; done
if [ ! -f $D/DAVIS_READY ]; then
  unzip -q -o davis2017_trainval_480p.zip -d $D/davis_raw && V=$D/davis_raw/DAVIS && mkdir -p $D/davis_val30/JPEGImages $D/davis_val30/Annotations && \
  for s in $(cat $V/ImageSets/2017/val.txt); do ln -sfn $V/JPEGImages/480p/$s $D/davis_val30/JPEGImages/$s; ln -sfn $V/Annotations/480p/$s $D/davis_val30/Annotations/$s; done && \
  echo "$(ls $D/davis_val30/JPEGImages | wc -l) seqs" > $D/DAVIS_READY
fi
while pgrep -u "$USER" -f "gdown.*1msjV2AAKROc" >/dev/null; do sleep 60; done
if [ ! -f $D/LVOS_READY ] && [ -f lvos_valid.zip ]; then
  unzip -q -o lvos_valid.zip -d $D/lvos_raw && R=$(dirname "$(find $D/lvos_raw -maxdepth 3 -type d -name JPEGImages | head -1)") && \
  ln -sfn "$R" $D/lvos_valid && echo "root=$R vids=$(ls $R/JPEGImages | wc -l)" > $D/LVOS_READY
fi
