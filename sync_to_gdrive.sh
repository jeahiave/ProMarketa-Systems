#!/bin/bash
# Sync ProMarketa output files to Google Drive archive
set -e

SRC="$HOME/mydev/ProMarketa-Systems"
DEST="gdrive:promarketa-archive"

echo "Starting archive sync at $(date)"

rclone copy "$SRC" "$DEST" \
  --include "04_raw_marketing_data.parquet" \
  --include "05_engineered_marketing_data.parquet" \
  --include "06_ProMarketa_intel_data.parquet" \
  --include "07_champion_churn_model.pkl" \
  --include "08_clv_regressor_model.pkl" \
  --progress

echo "Archive sync complete at $(date)"
