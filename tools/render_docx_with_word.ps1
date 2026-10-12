param(
    [Parameter(Mandatory = $true)] [string] $InputDocx,
    [Parameter(Mandatory = $true)] [string] $OutputDir,
    [Parameter(Mandatory = $true)] [string] $PdfToPpm
)

$ErrorActionPreference = 'Stop'
$resolvedInput = (Resolve-Path -LiteralPath $InputDocx).Path
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path -LiteralPath $resolvedOutput)) {
    New-Item -ItemType Directory -Path $resolvedOutput | Out-Null
}

$pdfPath = Join-Path $resolvedOutput 'patent-draft.pdf'
$pagePrefix = Join-Path $resolvedOutput 'page'
$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($resolvedInput, $false, $true)
    $document.ExportAsFixedFormat($pdfPath, 17)
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

& $PdfToPpm -png -r 150 $pdfPath $pagePrefix
if ($LASTEXITCODE -ne 0) {
    throw "pdftoppm failed with exit code $LASTEXITCODE"
}

Get-ChildItem -LiteralPath $resolvedOutput -Filter 'page-*.png' | Sort-Object Name | Select-Object -ExpandProperty FullName
