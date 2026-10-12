$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Source = Join-Path $ProjectRoot "bda_benchmark\dataset\splitname\standard_ML\test_set.txt"
$Target = Join-Path $ProjectRoot "bda_benchmark\dataset\splitname\standard_ML\test_set_20.txt"

$Lines = @(Get-Content -LiteralPath $Source | Select-Object -First 20)
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($Target, $Lines, $Utf8NoBom)
Write-Host "Wrote $($Lines.Count) entries to: $Target"
