# Podvia Windows 可重复打包脚本（阶段 7）
# 用法：在仓库根目录执行  powershell -ExecutionPolicy Bypass -File packaging/build.ps1
# 要求：Windows 10/11 + 项目 .venv（脚本会自动安装 pyinstaller 到该虚拟环境）
param(
    [switch]$SkipTests,
    [switch]$SkipPipInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "找不到 .venv\Scripts\python.exe，请先重建虚拟环境。" }

Write-Host "== [1/5] 安全检查：.env / API Key 不得进入安装包 ==" -ForegroundColor Cyan
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    $lines = Get-Content $envFile | Where-Object { $_ -match "DASHSCOPE_API_KEY\s*=\s*.+" }
    if ($lines) {
        throw ".env 中存在非空 DASHSCOPE_API_KEY。打包前请清空（GUI 会在用户目录另存配置）。"
    }
    Write-Host "   .env 存在但 Key 为空，继续（该文件不会被打入包内）。" -ForegroundColor Yellow
}

Write-Host "== [2/5] FFmpeg/ffprobe 分发策略 ==" -ForegroundColor Cyan
$ff = Join-Path $root "tools\bin\ffmpeg.exe"
$fp = Join-Path $root "tools\bin\ffprobe.exe"
if ((Test-Path $ff) -and (Test-Path $fp)) {
    Write-Host "   检测到 tools/bin，FFmpeg/ffprobe 将随包分发。" -ForegroundColor Green
} else {
    Write-Host "   未发现 tools/bin/ffmpeg.exe+ffprobe.exe：安装包不内置 FFmpeg。" -ForegroundColor Yellow
    Write-Host "   合并功能需要用户自行安装 FFmpeg（PATH）或放入 %APPDATA%\Podvia\tools\bin。" -ForegroundColor Yellow
}

if (-not $SkipPipInstall) {
    Write-Host "== [3/5] 安装/更新 PyInstaller（仅打包依赖，不写入 requirements.txt）==" -ForegroundColor Cyan
    & $py -m pip install --disable-pip-version-check pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 安装失败。" }
}

if (-not $SkipTests) {
    Write-Host "== [4/5] 封网跑全部 mock 单测 ==" -ForegroundColor Cyan
    $env:HTTPS_PROXY = "http://127.0.0.1:9"
    $env:HTTP_PROXY = "http://127.0.0.1:9"
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "单元测试未通过，已中止打包。" }
}

Write-Host "== [5/5] PyInstaller 构建 ==" -ForegroundColor Cyan
& $py -m PyInstaller --noconfirm --clean (Join-Path $root "packaging\podvia.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败。" }

$dist = Join-Path $root "dist\Podvia"
Write-Host "== 产物安全复核 ==" -ForegroundColor Cyan
if (Test-Path (Join-Path $dist ".env")) { throw "产物中包含 .env，打包失败。" }
$leaks = Get-ChildItem -Path $dist -Recurse -Include "*.env","*.key" -ErrorAction SilentlyContinue
if ($leaks) { throw "产物中发现疑似密钥文件：$($leaks.FullName -join ', ')" }
if (-not (Test-Path (Join-Path $dist "_internal\config.yaml"))) { throw "产物缺少 config.yaml 模板。" }
$exe = Join-Path $dist "Podvia.exe"
if (-not (Test-Path $exe)) { throw "产物缺少 Podvia.exe。" }

Write-Host ""
Write-Host "打包完成：$dist" -ForegroundColor Green
Write-Host "冒烟验证（可复制到任意中文/空格路径目录）：" -ForegroundColor Green
Write-Host "  $exe --help"
Write-Host "  $exe gui"
Write-Host "首次运行会在 %APPDATA%\Podvia 初始化 config.yaml 与空 .env。"
