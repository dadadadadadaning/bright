param(
    [Parameter(Mandatory = $true)] [string] $InputDocx,
    [Parameter(Mandatory = $true)] [string] $EquationJson,
    [int] $MaxCount = 0
)

$ErrorActionPreference = 'Stop'
$resolvedDocx = (Resolve-Path -LiteralPath $InputDocx).Path
$resolvedJson = (Resolve-Path -LiteralPath $EquationJson).Path
$equations = Get-Content -LiteralPath $resolvedJson -Raw -Encoding UTF8 | ConvertFrom-Json

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($resolvedDocx, $false, $false)

    $convertedCount = 0
    foreach ($item in $equations) {
        if ($MaxCount -gt 0 -and $convertedCount -ge $MaxCount) {
            break
        }
        $targetRange = $document.Content.Duplicate
        $find = $targetRange.Find
        $find.ClearFormatting()
        $find.Text = [string]$item.placeholder
        $find.Forward = $true
        $find.Wrap = 0
        if (-not $find.Execute()) {
            throw "Equation placeholder not found: $($item.placeholder)"
        }

        $linear = [string]$item.linear
        $targetRange.Text = $linear
        $targetRange.ParagraphFormat.Alignment = 1
        $targetRange.ParagraphFormat.FirstLineIndent = 0
        $targetRange.ParagraphFormat.LeftIndent = 0
        $targetRange.ParagraphFormat.RightIndent = 0
        $targetRange.Font.Name = 'Cambria Math'
        $targetRange.Font.NameFarEast = '宋体'
        $targetRange.Font.Size = 12

        [void]$document.OMaths.Add($targetRange)
        $mathObject = $targetRange.OMaths.Item(1)
        $mathObject.BuildUp()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($mathObject)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($targetRange)
        $convertedCount++
    }

    $document.Save()
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
