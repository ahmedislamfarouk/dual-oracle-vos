#!/bin/bash
# Thread-aware queue. Starts the next ready job only when account threads < ACC_MAX and < MAXJOBS X2 jobs run.
# A job is ready when its precondition file exists ("-" = none). Order in x3_jobs.txt = priority.
cd $HOME/ow_vis/experiments
Q=$HOME/ow_vis/experiments/x3_jobs.txt; LOG=$HOME/ow_vis/logs/x2/x3_sched.log; DONE=$HOME/ow_vis/logs/x2/x3_launched.txt
touch $DONE; MAXJOBS=3; ACC_MAX=390; GPUS=(4 2 3); gi=0
BASE="X2JOB=1 PYTORCH_NVML_BASED_CUDA_CHECK=0 ALLOC_CONF=none TMPDIR=$HOME/ow_vis/tmpdir OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_THREADS=1 USE_TF=0 SAM2_SIZE=large"
while true; do
  pending=$(grep -v '^#' $Q | cut -d'|' -f1 | grep -vxF -f $DONE | wc -l)
  [ "$pending" -eq 0 ] && { echo "$(date '+%F %T') queue empty, scheduler exits" >> $LOG; exit 0; }
  T=$(ps -L -u "$USER" --no-headers | wc -l); N=0
  for p in $(pgrep -u "$USER" -f "python3 (tide_x3|c4_x3|tide_ext|c4_multiseed).py"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx 'X2JOB=1' && N=$((N+1)); done
  launched=0
  if [ "$T" -lt "$ACC_MAX" ] && [ "$N" -lt "$MAXJOBS" ]; then
    while IFS='|' read -r TAG SCRIPT PRE EXTRA; do
      case "$TAG" in \#*|"") continue;; esac
      grep -qxF "$TAG" $DONE && continue
      [ "$PRE" != "-" ] && [ ! -f "$PRE" ] && continue
      G=${GPUS[$((gi % 3))]}; gi=$((gi+1))
      nohup env $BASE $EXTRA CUDA_VISIBLE_DEVICES=$G TAG=$TAG python3 $SCRIPT > $HOME/ow_vis/logs/x2/$TAG.log 2>&1 < /dev/null &
      echo "$TAG" >> $DONE; echo "$(date '+%F %T') launched $TAG on GPU $G (account=$T, x2 jobs before=$N)" >> $LOG
      launched=1; break
    done < <(grep -v '^#' $Q)
  fi
  if [ $launched -eq 1 ]; then sleep 120; else echo "$(date '+%F %T') waiting (account=$T, x2 jobs=$N, pending=$pending)" >> $LOG; sleep 300; fi
done
