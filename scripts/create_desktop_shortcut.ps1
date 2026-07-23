param(
    [Parameter(Mandatory = $true)]
    [string]$PythonwPath,

    [string]$ProjectDirectory = (Split-Path -Parent $PSScriptRoot),

    [string]$ShortcutPath = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'BioInsAlgo.lnk')
)

$resolvedPythonw = (Resolve-Path -LiteralPath $PythonwPath).Path
$resolvedProject = (Resolve-Path -LiteralPath $ProjectDirectory).Path
$resolvedLauncher = (Resolve-Path -LiteralPath (Join-Path $resolvedProject 'scripts\launch_gui.pyw')).Path

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)
$shortcut.TargetPath = $resolvedPythonw
$shortcut.Arguments = '"' + $resolvedLauncher + '"'
$shortcut.WorkingDirectory = $resolvedProject
$shortcut.Description = 'BioInsAlgo soybean root skeletonization'
$shortcut.IconLocation = "$resolvedPythonw,0"
$shortcut.Save()

Write-Output $ShortcutPath
