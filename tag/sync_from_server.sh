#!/bin/bash
# Periodically sync shard data files from server to local machine.
# Usage: bash sync_from_server.sh [interval_seconds]  (default: 300)

SERVER="cc@192.5.87.187"
KEY="$HOME/.ssh/id_rsa_chameleon"
REMOTE_DIR="~/Flickr30K-CFQ/tag/data"
LOCAL_DIR="$(dirname "$0")/data"
INTERVAL="${1:-300}"

mkdir -p "$LOCAL_DIR"

echo "Syncing every ${INTERVAL}s from ${SERVER}:${REMOTE_DIR} → ${LOCAL_DIR}"
echo "Press Ctrl+C to stop."
echo ""

while true; do
    TIMESTAMP=$(date '+%H:%M:%S')
    rsync -az -e "ssh -i $KEY" \
        "${SERVER}:${REMOTE_DIR}/*_shard*.json" \
        "$LOCAL_DIR/" 2>/dev/null

    # Count total processed images across all shards
    TOTAL=0
    for f in "$LOCAL_DIR"/*tag*_shard*.json; do
        [ -f "$f" ] || continue
        COUNT=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null)
        TOTAL=$((TOTAL + ${COUNT:-0}))
    done

    echo "[$TIMESTAMP] Synced. Total tagged so far: ${TOTAL} / 31783"
    sleep "$INTERVAL"
done
