Add-Type -AssemblyName PresentationCore

$sourceDir = "D:\VLM\PSNA BUS"
$targetDir = "D:\VLM\PSNA_Buses_JPG"

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
}

$heicFiles = Get-ChildItem -Path $sourceDir -Filter *.heic
$successCount = 0

foreach ($file in $heicFiles) {
    $jpgPath = Join-Path $targetDir ($file.BaseName + ".jpg")
    try {
        $stream = New-Object System.IO.FileStream($file.FullName, [System.IO.FileMode]::Open)
        $decoder = [System.Windows.Media.Imaging.BitmapDecoder]::Create($stream, [System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat, [System.Windows.Media.Imaging.BitmapCacheOption]::Default)
        
        $encoder = New-Object System.Windows.Media.Imaging.JpegBitmapEncoder
        $encoder.Frames.Add($decoder.Frames[0])
        
        $outStream = New-Object System.IO.FileStream($jpgPath, [System.IO.FileMode]::Create)
        $encoder.Save($outStream)
        
        $outStream.Close()
        $stream.Close()
        $successCount++
        Write-Host "Converted: $($file.Name)"
    } catch {
        Write-Host "Failed to convert $($file.Name): $($_.Exception.Message)"
    }
}
Write-Host "Total converted: $successCount"
