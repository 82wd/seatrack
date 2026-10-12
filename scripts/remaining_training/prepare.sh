#!/usr/bin/env bash
set -euo pipefail
TASK=$1
BASE=/nas/liziyan103/seatrack
RUN=$BASE/training_runs/20261011_remaining
mkdir -p "$RUN/$TASK"
exec > >(tee -a "$RUN/$TASK/process.log") 2>&1
exec 8>"$RUN/$TASK/task.lock"
flock -n 8 || exit 0
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "FAILED exit=%s time=%s\n" "$rc" "$(date -Is)" > "$RUN/$TASK/status.txt"; fi' EXIT
printf 'PREPARING %s\n' "$(date -Is)" > "$RUN/$TASK/status.txt"
if [ ! -f "$RUN/$TASK/extraction.complete" ]; then
  if [ "$TASK" = rgbt ]; then
    cat "$BASE"/datasets/lasher/lasher.tar.gz.part.a? | nice -n 10 tar -xzf - -C "$BASE/datasets" --wildcards 'lasher/trainingset/*'
  elif [ "$TASK" = rgbe ]; then
    nice -n 10 "$BASE/tools/7zip/7zz" x -y -bsp0 -mmt=2 \
      "$BASE/datasets/visevent/VisEvent_dataset/VisEvent_train.zip" \
      -o"$BASE/datasets/visevent" '-xr!*.aedat4' '-xr!*_events.txt'
  else
    exit 2
  fi
  date -Is > "$RUN/$TASK/extraction.complete"
fi
printf 'WAITING_CONFIGURATION %s\n' "$(date -Is)" > "$RUN/$TASK/status.txt"
while [ ! -f "$RUN/launch_ready" ]; do sleep 30; done
/nas/liziyan103/anaconda3/envs/seatrack/bin/python "$RUN/scripts/run_task.py" "$TASK"
