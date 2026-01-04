#!/usr/bin/env bash

BATCH_DIR="data/batches"
PROCESSED_DIR="data/processed"
INTERVAL=$((60 * 60 * 24 * 30))  # 1 month in seconds
 
while true; do 
    echo "Starting preprocessing at $(date)..."

    python -c "from pipelines.prepare_batches import prepare_batches; prepare_batches()"
    python -c "from main import preprocess_all_batches; preprocess_all_batches('$BATCH_DIR', '$PROCESSED_DIR')"

    echo "Done at $(date). Sleeping for $INTERVAL seconds..."
    sleep $INTERVAL
done