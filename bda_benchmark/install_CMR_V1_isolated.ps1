param(
    [string]$ProjectRoot = 'D:\software\dowmload\BRIGHT-master'
)

$ErrorActionPreference = 'Stop'
$PatchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$Files = @(
    'model\cmr_v1\__init__.py',
    'model\cmr_v1\set_matcher.py',
    'script\standard_ML\cmr_v1\__init__.py',
    'script\standard_ML\cmr_v1\audit_top20_complement.py',
    'script\standard_ML\cmr_v1\build_token_cache.py',
    'script\standard_ML\cmr_v1\audit_set_verifier.py',
    'tests\cmr_v1\__init__.py',
    'tests\cmr_v1\test_set_matcher.py',
    'configs\cmr_v1\paths.example.env',
    'run\run_CMR_V1_isolated.sh',
    'docs\CMR_V1_ISOLATED_ZH.md',
    'docs\CMR_V1_ISOLATION_MANIFEST.txt'
)

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project root does not exist: $ProjectRoot"
}

$Conflicts = @()
foreach ($rel in $Files) {
    $dst = Join-Path $ProjectRoot $rel
    if (Test-Path -LiteralPath $dst) {
        $Conflicts += $dst
    }
}

if ($Conflicts.Count -gt 0) {
    Write-Host 'CMR-V1 isolated install aborted. Existing V1 targets were found:' -ForegroundColor Yellow
    $Conflicts | ForEach-Object { Write-Host "  $_" }
    throw 'No files were copied. Move/remove the existing V1 files or use the already-installed version.'
}

foreach ($rel in $Files) {
    $src = Join-Path $PatchRoot $rel
    $dst = Join-Path $ProjectRoot $rel
    $dstDir = Split-Path -Parent $dst
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst
    Write-Host "ADDED  $rel"
}

Write-Host ''
Write-Host 'CMR-V1 isolated integration completed.' -ForegroundColor Green
Write-Host 'No pre-existing CMR source file was overwritten.'
Write-Host 'Next: inspect docs\CMR_V1_ISOLATED_ZH.md'
