param([string]$Checkout = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$Checkout = (Resolve-Path -LiteralPath $Checkout).Path
$interpreter = Join-Path $Checkout '.venv-gpu\Scripts\pythonw.exe'
$launcher = Join-Path $Checkout 'scripts\launch_gpu.py'
if (-not (Test-Path -LiteralPath $interpreter) -or -not (Test-Path -LiteralPath $launcher)) {
    throw 'Create .venv-gpu and install GPU dependencies before creating the shortcut.'
}
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'SoyRootArchitect GPU.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $interpreter
$shortcut.Arguments = '"' + $launcher + '"'
$shortcut.WorkingDirectory = $Checkout
$shortcut.Description = 'Isolated SoyRootArchitect GPU-version; CUDA required'
$shortcut.Save()
$verify = $shell.CreateShortcut($shortcutPath)
if ($verify.TargetPath -ne $interpreter -or $verify.WorkingDirectory -ne $Checkout -or
    $verify.Arguments -ne ('"' + $launcher + '"')) {
    throw 'Shortcut verification failed.'
}
Write-Output $shortcutPath
