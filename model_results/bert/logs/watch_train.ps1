$ErrorActionPreference = "Continue"
Write-Output ("WATCH_START " + (Get-Date -Format o))

$targetPids = @(13160, 24976)
Write-Output ("watching_pids=" + ($targetPids -join ","))

while ($true) {
  $alive = @()
  foreach ($procId in $targetPids) {
    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
      $alive += $procId
    }
  }

  # Also discover any still-running experiment python
  $extra = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "run_all_experiments|train_bert|extract_embeddings|train_bert_cnn2d" } |
    Select-Object -ExpandProperty ProcessId
  foreach ($procId in $extra) {
    if ($alive -notcontains $procId) { $alive += $procId }
  }

  Write-Output ("heartbeat alive=" + (($alive -join ",") + " t=" + (Get-Date -Format HH:mm:ss)))

  $trainLog = "f:\final-version\independent_ml_workflow\models\bert\logs\training.log"
  $doneMsg = $false
  if (Test-Path $trainLog) {
    $tail = Get-Content $trainLog -Tail 100 -ErrorAction SilentlyContinue | Out-String
    if ($tail -match "All requested experiments finished") { $doneMsg = $true }
  }

  if (($alive.Count -eq 0) -or $doneMsg) {
    $status = if ($doneMsg) { "SUCCESS" } else { "PROCESS_EXITED" }
    $donePath = "f:\final-version\independent_ml_workflow\models\bert\logs\TRAINING_DONE.txt"
    $lines = @()
    $lines += "status=$status"
    $lines += ("time=" + (Get-Date -Format o))
    $lines += "--- training.log tail ---"
    if (Test-Path $trainLog) { $lines += Get-Content $trainLog -Tail 50 }
    $lines | Set-Content -Path $donePath -Encoding utf8
    Write-Output ("AGENT_LOOP_WAKE_bert_train {`"prompt`":`"BERT training finished. Read f:\\final-version\\independent_ml_workflow\\models\\bert\\logs\\TRAINING_DONE.txt and results/; summarize status and key metrics; notify user in Chinese that training is done.`",`"status`":`"$status`"}")
    break
  }

  Start-Sleep -Seconds 90
}

Write-Output ("WATCH_END " + (Get-Date -Format o))
