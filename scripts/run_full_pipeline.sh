#!/bin/bash
# Full Data Preparation Pipeline (Improved Alignment)
# Runs all steps sequentially and logs output

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/data/processed/pipeline.log"

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "========================================" | tee -a "$LOG_FILE"
echo "TuvaLLM Data Pipeline Started: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Step 1: Transcribe with MMS-Samoan (already done if restarting, script skips existing)
echo "[$(date)] Step 1: Transcribing audio with MMS..." | tee -a "$LOG_FILE"
python scripts/transcribe_with_mms.py 2>&1 | tee -a "$LOG_FILE"

# Step 2: Transcribe with Whisper English (for English loanwords/names)
echo "[$(date)] Step 2: Transcribing with Whisper English..." | tee -a "$LOG_FILE"
python scripts/transcribe_whisper_english.py 2>&1 | tee -a "$LOG_FILE"

# Step 3: Merge anchor sources (MMS + Whisper)
echo "[$(date)] Step 3: Merging anchor sources..." | tee -a "$LOG_FILE"
python scripts/merge_anchor_sources.py 2>&1 | tee -a "$LOG_FILE"

# Step 4: Sequence-based alignment (LCS matching)
echo "[$(date)] Step 4: Running sequence alignment..." | tee -a "$LOG_FILE"
python scripts/sequence_align.py 2>&1 | tee -a "$LOG_FILE"

# Step 5: Generate confident segments
echo "[$(date)] Step 5: Generating confident segments..." | tee -a "$LOG_FILE"
python scripts/anchor_align.py 2>&1 | tee -a "$LOG_FILE"

# Step 6: Verify alignment quality
echo "[$(date)] Step 6: Generating verification samples..." | tee -a "$LOG_FILE"
python scripts/verify_alignment.py 2>&1 | tee -a "$LOG_FILE"

# Step 7: Prepare final dataset
echo "[$(date)] Step 7: Preparing dataset..." | tee -a "$LOG_FILE"
python scripts/prepare_dataset.py 2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "Pipeline Complete: $(date)" | tee -a "$LOG_FILE"
echo "Dataset ready in: $PROJECT_DIR/data/processed/dataset/" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Summary
echo ""
echo "Next steps:"
echo "  1. Review verification samples in: data/processed/verification_samples/"
echo "  2. Check quality_report.json and verification_checklist.md"
echo "  3. If quality looks good, run training"
echo ""

# Optional: Send notification (macOS)
osascript -e 'display notification "TuvaLLM dataset ready!" with title "Pipeline Complete"' 2>/dev/null || true
