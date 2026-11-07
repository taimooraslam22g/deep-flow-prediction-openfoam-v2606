# Train and test the U-Net several times with different random seeds, one after another.
#
# Usage (from C:\Users\boy10\Project2):
#     powershell -ExecutionPolicy Bypass -File scripts\run_seeds.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\run_seeds.ps1 -Seeds 1,2,3
#
# Each run uses its own file prefix (seed1_, seed2_, ...), so nothing is overwritten:
#   seedN_modelG     trained model
#   seedN_L1.txt     training loss per epoch
#   seedN_L1val.txt  validation loss per epoch
#   seedN_testout.txt  test results on the 90 test airfoils
#   seedN_train.log  everything the training printed

param(
    [int[]]$Seeds = @(1, 2, 3, 4, 5)
)

$python = "C:\Users\boy10\Project2\.venv\Scripts\python.exe"
$train  = "C:\Users\boy10\Project2\Deep-Flow-Prediction\train"
Set-Location $train

foreach ($seed in $Seeds) {
    $prefix = "seed${seed}_"
    if (Test-Path "${prefix}testout.txt") {
        Write-Host "Seed $seed already done, skipping."
        continue
    }
    $start = Get-Date
    Write-Host "=== Seed $seed : training started $($start.ToString('HH:mm')) ==="
    & $python runTrainCpu_expo5.py $prefix $seed *> "${prefix}train.log"
    if ($LASTEXITCODE -ne 0) { Write-Host "Training failed for seed $seed - see ${prefix}train.log"; exit 1 }

    Write-Host "    testing..."
    & $python runTestCpu_expo5.py $prefix *> $null
    $summary = Select-String -Path "${prefix}testout.txt" -Pattern "Loss percentage" | Select-Object -Last 1
    $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    Write-Host "    done in $mins min: $($summary.Line.Trim())"
}

Write-Host "`nAll seeds finished. Summary:"
& $python C:\Users\boy10\Project2\scripts\summarize_seeds.py $train
