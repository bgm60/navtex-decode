$logDir = "D:\Radio\Logs\518kHz"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

& python navtex_decode.py 518kHz 2>&1 | ForEach-Object {
    $now = (Get-Date).ToUniversalTime()
    $timestamp = $now.ToString("yyyy-MM-dd HH:mm:ss")
    $logFile = Join-Path $logDir ("518-" + $now.ToString("yyyy-MM-dd") + ".log")
    
    # Writes to the file (appends and rotates daily based on the date in the filename)
    "[$timestamp] $_" | Out-File -FilePath $logFile -Append -Encoding utf8
}