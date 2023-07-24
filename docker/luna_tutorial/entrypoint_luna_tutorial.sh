#!/bin/bash
#
# source .bashrc to make sure conda is set up, along with other things we'd
# expect to find if we opened a terminal in the container.

set -o pipefail

source /usr/local/bin/_activate_current_env.sh

LUNA_HOME="$HOME/vmount"
LOG_DIR="$LUNA_HOME/logs"

SCHEDULER_LOG="$LOG_DIR/dask_scheduler.log"
WORKER_LOG="$LOG_DIR/dask_worker.log"
DASK_ADDR_FILE="$LUNA_HOME/.dask_addr"

dask scheduler --port 8786 > "$SCHEDULER_LOG" 2>&1 &
sleep 1

SCHEDULER_ADDR=""
while [[ -z $SCHEDULER_ADDR ]]; do
    echo "[waiting for scheduler to spin up...]"
    sleep 1
    SCHEDULER_ADDR=$(grep 'Scheduler at:.*tcp:' "$SCHEDULER_LOG" | sed 's!.* tcp:!tcp:!')
done
echo $SCHEDULER_ADDR > "$DASK_ADDR_FILE"

dask worker "$SCHEDULER_ADDR" > "$WORKER_LOG" 2>&1 &
echo "Dask cluster is now running"

jupyter notebook --ip 0.0.0.0 --notebook-dir="$HOME/vmount" > ~/vmount/logs/tutorial.log 2>&1 
