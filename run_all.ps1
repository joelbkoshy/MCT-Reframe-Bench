param([string]$Python = "python")

# Runs every stage in order, detached from the editor; a failed stage stops the chain.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Keep the machine awake while this window runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); no admin needed.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
$log = Join-Path $PSScriptRoot "results\run_log.txt"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

foreach ($stage in @("fetch", "build", "calibrate", "validate", "generate", "judge", "analyse", "figures", "architecture")) {
    "[$(Get-Date -Format s)] === $stage ===" | Out-File -Append -Encoding utf8 $log
    cmd /c "`"$Python`" -X utf8 run.py $stage >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        "[$(Get-Date -Format s)] $stage FAILED (exit $LASTEXITCODE); rerun this script to resume" | Out-File -Append -Encoding utf8 $log
        exit $LASTEXITCODE
    }
}
"[$(Get-Date -Format s)] === done ===" | Out-File -Append -Encoding utf8 $log
