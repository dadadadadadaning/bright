[CmdletBinding()]
param(
    [switch]$SmokeTest,
    [switch]$Overwrite,
    [int]$MaxCheckpoints = 0,
    [ValidateSet(1, 2, 3)]
    [int]$PriorityMax = 3,
    [string]$PythonExe = "",
    [string]$TestList = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$FullList = Join-Path $ProjectRoot "bda_benchmark\dataset\splitname\standard_ML\test_set.txt"
$SubsetList = Join-Path $ProjectRoot "bda_benchmark\dataset\splitname\standard_ML\test_set_20.txt"

if ($SmokeTest) {
    if (-not (Test-Path -LiteralPath $SubsetList)) {
        & (Join-Path $PSScriptRoot "make_test_subset_20.ps1")
    }
    $TestList = $SubsetList
    $ResultsRoot = Join-Path $ProjectRoot "eval_results_official_local_cpu_smoke20"
    $AnalysisRoot = Join-Path $ProjectRoot "eval_analysis_cpu_smoke20"
}
elseif ([string]::IsNullOrWhiteSpace($TestList)) {
    $TestList = $FullList
    $ResultsRoot = Join-Path $ProjectRoot "eval_results_official_local_cpu"
    $AnalysisRoot = Join-Path $ProjectRoot "eval_analysis_cpu"
}
else {
    $ResultsRoot = Join-Path $ProjectRoot "eval_results_official_local_cpu"
    $AnalysisRoot = Join-Path $ProjectRoot "eval_analysis_cpu"
}

$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:MPLCONFIGDIR = Join-Path $env:TEMP "bright_mpl_cache"

Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $CondaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $CondaPython) {
            $PythonExe = $CondaPython
        }
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

& $PythonExe -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    $KnownBrightPython = "D:\anaconda3\envs\bright\python.exe"
    if ((Test-Path -LiteralPath $KnownBrightPython) -and ($PythonExe -ne $KnownBrightPython)) {
        Write-Warning "Current Python cannot import torch; falling back to $KnownBrightPython"
        $PythonExe = $KnownBrightPython
        & $PythonExe -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
    }
}
if ($LASTEXITCODE -ne 0) {
    throw "No usable Python with torch was found. Activate conda env 'bright' or pass -PythonExe."
}
Write-Host "Using Python: $PythonExe"

$RunnerArgs = @(
    ".\scripts\local_eval\local_eval_all_official.py",
    "--test-list", $TestList,
    "--results-root", $ResultsRoot,
    "--analysis-root", $AnalysisRoot,
    "--python", $PythonExe,
    "--priority-max", $PriorityMax
)
if ($Overwrite) {
    $RunnerArgs += "--overwrite"
}
if ($MaxCheckpoints -gt 0) {
    $RunnerArgs += @("--max-checkpoints", $MaxCheckpoints)
}
& $PythonExe @RunnerArgs
$RunExitCode = $LASTEXITCODE

& $PythonExe .\scripts\local_eval\summarize_local_eval_results.py --results-root $ResultsRoot --analysis-root $AnalysisRoot
$SummaryExitCode = $LASTEXITCODE

if ($RunExitCode -ne 0) {
    throw "Local evaluator runner failed with exit code $RunExitCode."
}
if ($SummaryExitCode -ne 0) {
    throw "Local result summarizer failed with exit code $SummaryExitCode."
}
