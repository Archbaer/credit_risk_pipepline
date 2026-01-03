BATCH_DIR="data/raw"
PROCESSED_DIR="data/processed"
INTERVAL=2629800 # Approximately one month in seconds


while true; do
    echo "Starting preprocessing at $(date)..."
    
    echo "Preparing batches..."
    python -c "
    from pipelines.prepare_batches import prepare_batches

    prepare_batches()
    "
    
    python -c "
    from main import preprocess_all_batches
    preprocess_all_batches('$BATCH_DIR', '$PROCESSED_DIR')
    "

    echo "Preprocessing done on batches from $BATCH_DIR"
    
    echo "Preprocessing completed at $(date). Waiting for next interval..."
    sleep $INTERVAL
done