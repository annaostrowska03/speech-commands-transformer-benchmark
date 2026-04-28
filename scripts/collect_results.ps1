$AnalysisDir = "outputs/analysis"

if (-not (Test-Path $AnalysisDir)) {
    New-Item -Path $AnalysisDir -ItemType Directory -Force | Out-Null
}

python src/reporting.py --outputs_dir outputs --analysis_dir outputs/analysis
if ($LASTEXITCODE -ne 0) {
    throw "Result collection failed."
}

Write-Host ""
Write-Host "Expected generated files:"
Write-Host "- outputs/analysis/leaderboard.csv"
Write-Host "- outputs/analysis/leaderboard.md"
Write-Host "- outputs/analysis/leaderboard.json"
Write-Host "- outputs/analysis/top_confusions.csv"
Write-Host "- outputs/analysis/unknown_silence.csv"
