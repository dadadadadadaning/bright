[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$TestList = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($TestList)) {
    $TestList = Join-Path $ProjectRoot "bda_benchmark\dataset\splitname\standard_ML\test_set.txt"
}

$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:MPLCONFIGDIR = Join-Path $env:TEMP "bright_mpl_cache"

Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($PythonExe) -and -not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
    $Candidate = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path -LiteralPath $Candidate) {
        $PythonExe = $Candidate
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

$PreviousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $PythonExe -c "import torch" 2>$null
$TorchImportExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorAction
if ($TorchImportExitCode -ne 0) {
    $KnownBrightPython = "D:\anaconda3\envs\bright\python.exe"
    if (Test-Path -LiteralPath $KnownBrightPython) {
        Write-Warning "Selected Python cannot import torch; using $KnownBrightPython"
        $PythonExe = $KnownBrightPython
    } else {
        throw "No usable Python with torch was found. Activate conda env 'bright' or pass -PythonExe."
    }
}
& $PythonExe .\scripts\local_eval\local_eval_all_official.py --dry-run --test-list $TestList --python $PythonExe
