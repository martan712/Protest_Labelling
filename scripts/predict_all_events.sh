#!/usr/bin/env bash
# Label the full event corpus with a trained checkpoint.
#
# Tuned for this machine from measured throughput: ~61 rows/s end to end at a
# 4.1GB peak, so ~50-60 minutes for 218k events. The read chunk is what matters
# most -- predict_csv length-sorts within each chunk before batching, so a wider
# chunk wastes less padding. Peak memory is dominated by the weights, not the
# activations, so a larger batch buys nothing here.
#
# Override any of these from the environment, e.g. BATCH_SIZE=64 ./scripts/predict_all_events.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-models/new_classifier/run-6000/seed-42/best}"
EVENTS="${EVENTS:-data/raw/events.csv}"
OUTPUT="${OUTPUT:-data/predictions/all_events_predictions.csv}"
BATCH_SIZE="${BATCH_SIZE:-32}"
READ_CHUNK_SIZE="${READ_CHUNK_SIZE:-8192}"
MAX_LENGTH="${MAX_LENGTH:-256}"
PYTHON="${PYTHON:-.venv/bin/python}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

for path in "$MODEL" "$EVENTS" "$PYTHON"; do
    if [[ ! -e "$path" ]]; then
        echo "error: $path not found" >&2
        exit 1
    fi
done

mkdir -p logs "$(dirname "$OUTPUT")"
LOG="${LOG:-logs/predict_all_events_$(date +%Y%m%d_%H%M%S).log}"

echo "model      $MODEL"
echo "events     $EVENTS ($(( $(wc -l < "$EVENTS") - 1 )) rows)"
echo "output     $OUTPUT"
echo "batch      $BATCH_SIZE | read chunk $READ_CHUNK_SIZE | max length $MAX_LENGTH"
echo "log        $LOG"
echo

START=$SECONDS
"$PYTHON" scripts/predict_events.py \
    --model "$MODEL" \
    --events "$EVENTS" \
    --output "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --read-chunk-size "$READ_CHUNK_SIZE" \
    --max-length "$MAX_LENGTH" \
    2>&1 | tee "$LOG"

ELAPSED=$(( SECONDS - START ))
ROWS=$(( $(wc -l < "$OUTPUT") - 1 ))
printf '\ndone: %d rows in %dm %02ds (%.1f rows/s) -> %s\n' \
    "$ROWS" "$(( ELAPSED / 60 ))" "$(( ELAPSED % 60 ))" \
    "$(awk "BEGIN {print $ROWS / ($ELAPSED ? $ELAPSED : 1)}")" "$OUTPUT"
