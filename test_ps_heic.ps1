Add-Type -AssemblyName System.Drawing
[System.Reflection.Assembly]::LoadWithPartialName("Windows.Graphics.Imaging") | Out-Null
Write-Host "Assembly loaded."
