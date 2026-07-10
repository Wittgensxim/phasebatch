param(
    [string]$LLVMDir = "E:/llvm/build/lib/cmake/llvm",
    [string]$BuildDir = "worker/build"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceDir = Join-Path $repoRoot "worker"
$resolvedBuildDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
$gxx = "C:/msys64/ucrt64/bin/g++.exe"

cmake -S $sourceDir -B $resolvedBuildDir -G Ninja `
    "-DLLVM_DIR=$LLVMDir" `
    "-DCMAKE_CXX_COMPILER=$gxx" `
    "-DCMAKE_BUILD_TYPE=Release"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
}

cmake --build $resolvedBuildDir --target phasebatch-worker
if ($LASTEXITCODE -ne 0) {
    throw "Worker build failed with exit code $LASTEXITCODE"
}

$worker = Join-Path $resolvedBuildDir "phasebatch-worker.exe"
if (-not (Test-Path -LiteralPath $worker)) {
    $worker = Join-Path $resolvedBuildDir "phasebatch-worker"
}
if (-not (Test-Path -LiteralPath $worker)) {
    throw "Worker binary was not produced under $resolvedBuildDir"
}

Write-Output $worker
