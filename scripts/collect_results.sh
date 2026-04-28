#!/usr/bin/env bash
set -e

mkdir -p outputs/analysis

python src/reporting.py --outputs_dir outputs --analysis_dir outputs/analysis

echo
echo "Expected generated files:"
echo "- outputs/analysis/leaderboard.csv"
echo "- outputs/analysis/leaderboard.md"
echo "- outputs/analysis/leaderboard.json"
echo "- outputs/analysis/top_confusions.csv"
echo "- outputs/analysis/unknown_silence.csv"
