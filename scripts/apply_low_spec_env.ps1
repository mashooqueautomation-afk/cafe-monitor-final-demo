$ErrorActionPreference = 'Stop'
$envPath = Join-Path (Split-Path -Parent $PSScriptRoot) '.env'

# These values are intended for the bundled dual-core / 8 GB RAM demo profile.
# Only performance-related keys are changed; camera URLs/passwords and other settings are preserved.
$values = [ordered]@{
    YOLO_MODEL = '/workspace/models/yolov8n.pt'
    YOLO_BACKEND = 'auto'
    YOLO_AUTO_DOWNLOAD = 'true'
    YOLO_IMGSZ = '640'
    YOLO_CONFIDENCE = '0.20'
    PERSON_DETECTION_IMGSZ = '640'
    PERSON_DETECTION_CONFIDENCE = '0.12'
    PERSON_TILED_INFERENCE = 'true'
    MAX_RESULT_FRAME_AGE_MS = '5000'
    YOLO_DEVICE = 'cpu'
    TORCH_NUM_THREADS = '1'
    TORCH_INTEROP_THREADS = '1'
    OPENCV_NUM_THREADS = '1'
    ANALYTICS_FPS = '0.2'
    ENTRANCE_FPS = '0.5'
    DEMO_PLAYBACK_FPS = '4'
    DEMO_USE_PROXIES = 'true'
    DEMO_PROXY_FPS = '4'
    DEMO_PROXY_WIDTH = '640'
    DEMO_PROXY_DIR = '/workspace/runtime/demo_proxy'
    FOCUS_MAX_CAMERAS = '1'
    FOCUS_ENTRANCE_FPS = '2'
    FOCUS_ANALYTICS_FPS = '0.2'
    DISPLAY_FPS = '1.5'
    JPEG_QUALITY = '50'
    MAX_STREAM_WIDTH = '640'
}

if (-not (Test-Path -LiteralPath $envPath)) {
    throw "Missing .env file: $envPath. Copy .env.example to .env first."
}

$lines = [System.Collections.Generic.List[string]]::new()
foreach ($line in [System.IO.File]::ReadAllLines($envPath)) { $lines.Add($line) }
$found = @{}
foreach ($key in $values.Keys) { $found[$key] = $false }
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\s*([A-Z0-9_]+)\s*=') {
        $key = $Matches[1]
        if ($values.Contains($key)) {
            $lines[$i] = "$key=$($values[$key])"
            $found[$key] = $true
        }
    }
}
foreach ($key in $values.Keys) {
    if (-not $found[$key]) { $lines.Add("$key=$($values[$key])") }
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($envPath, $lines, $utf8NoBom)
Write-Host 'Applied low-spec performance settings; RTSP URLs and unrelated settings were preserved.' -ForegroundColor Green
