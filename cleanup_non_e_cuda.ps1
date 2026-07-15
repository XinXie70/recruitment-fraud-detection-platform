$ErrorActionPreference = "Continue"

$logPath = Join-Path $PSScriptRoot "cleanup_non_e_cuda.log"
Start-Transcript -Path $logPath -Force | Out-Null

function Remove-ExactPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if ($fullPath.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "SKIP E drive path: $fullPath"
        return
    }

    if (Test-Path -LiteralPath $fullPath) {
        Write-Host "Removing: $fullPath"
        takeown.exe /F $fullPath /R /D Y | Out-Null
        icacls.exe $fullPath /grant "Administrators:(OI)(CI)F" /T /C | Out-Null
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    } else {
        Write-Host "Already absent: $fullPath"
    }
}

$python = "D:\Anaconda3\python.exe"
$condaMeta = "D:\Anaconda3\conda-meta"

Get-Process | Where-Object {
    ($_.ProcessName -match "python|conda") -and ($_.Path -like "D:\Anaconda3\*")
} | Stop-Process -Force

if (Test-Path -LiteralPath $python) {
    & $python -m pip uninstall -y torch torchvision torchaudio
}

if (Test-Path -LiteralPath $condaMeta) {
    $cudaMetaFiles = Get-ChildItem -LiteralPath $condaMeta -Filter "*.json" | Where-Object {
        $packageName = $_.BaseName -replace "-[0-9].*$", ""
        $packageName -match "^(cuda|libcu|libnv|nsight)"
    }

    foreach ($metaFile in $cudaMetaFiles) {
        Write-Host "Removing conda package from manifest: $($metaFile.Name)"
        $package = Get-Content -LiteralPath $metaFile.FullName -Raw | ConvertFrom-Json
        foreach ($relativeFile in $package.files) {
            $target = Join-Path "D:\Anaconda3" $relativeFile
            if ($target.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
                Write-Host "SKIP E drive path: $target"
                continue
            }
            if (Test-Path -LiteralPath $target -PathType Leaf) {
                Remove-Item -LiteralPath $target -Force
            }
        }

        $package.files |
            ForEach-Object { Split-Path -Path (Join-Path "D:\Anaconda3" $_) -Parent } |
            Sort-Object Length -Descending -Unique |
            ForEach-Object {
                if ($_.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
                    return
                }
                if ((Test-Path -LiteralPath $_ -PathType Container) -and -not (Get-ChildItem -LiteralPath $_ -Force)) {
                    Remove-Item -LiteralPath $_ -Force
                }
            }

        Remove-Item -LiteralPath $metaFile.FullName -Force
    }
}

$pathsToRemove = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    "D:\Anaconda3\Lib\site-packages\functorch",
    "D:\Anaconda3\Lib\site-packages\torch",
    "D:\Anaconda3\Lib\site-packages\torch-2.6.0+cu124.dist-info",
    "D:\Anaconda3\Lib\site-packages\torchgen",
    "D:\Anaconda3\Lib\site-packages\torchvision",
    "D:\Anaconda3\Lib\site-packages\torchvision-0.21.0+cu124.dist-info"
)

foreach ($path in $pathsToRemove) {
    Remove-ExactPath -Path $path
}

Write-Host "Remaining nvcc locations:"
where.exe nvcc

if (Test-Path -LiteralPath "D:\Anaconda3\Scripts\conda.exe") {
    Write-Host "Remaining CUDA-related conda packages:"
    & "D:\Anaconda3\Scripts\conda.exe" list | Select-String -Pattern "^(cuda|libcu|libnv|nsight|torch|torchvision|torchaudio|triton|nvidia)"
}

Stop-Transcript | Out-Null
