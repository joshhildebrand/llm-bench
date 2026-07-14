# RAM watchdog for risky model loads. Run in the background while loading/benching
# a model that might exhaust system memory (e.g. a 63 GB hybrid load on a 64 GB box):
#
#   powershell -File ramguard.ps1 -MinFreeCommitGB 6 -LogPath ramguard.log
#
# What actually hard-locks Windows is COMMIT exhaustion (RAM+pagefile): CUDA sysmem
# fallback and mlock'd weights are committed and non-reclaimable. Low *physical*
# free memory alone is normal during mmap streaming — clean mapped pages are
# reclaimable page cache and Windows trims them under pressure — so the physical
# floor is low and requires two consecutive readings before acting.
# Escalation: `lms unload --all` -> recheck -> kill the fattest LM Studio process.
param(
    [double]$MinFreeCommitGB = 6.0,   # primary trigger: available commit charge
    [double]$MinFreePhysGB = 0,       # 0 = disabled. Low free physical RAM is NORMAL during
                                      # mmap prefill (clean read-only pages, reclaimed for free)
                                      # and false-triggers; only enable for non-mmap loads.
    [int]$PollSeconds = 2,
    [string]$LogPath = "ramguard.log"
)

function Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Get-Mem {
    $os = Get-CimInstance Win32_OperatingSystem
    @{ Phys = $os.FreePhysicalMemory / 1MB; Commit = $os.FreeVirtualMemory / 1MB }
}

Log "ramguard started: commitFloor=${MinFreeCommitGB}GB physFloor=${MinFreePhysGB}GB poll=${PollSeconds}s"
$physStrikes = 0
while ($true) {
    $m = Get-Mem
    $commitLow = $m.Commit -lt $MinFreeCommitGB
    if ($MinFreePhysGB -gt 0 -and $m.Phys -lt $MinFreePhysGB) { $physStrikes++ } else { $physStrikes = 0 }

    if ($commitLow -or $physStrikes -ge 2) {
        Log ("TRIGGER: freePhys={0:N1}GB freeCommit={1:N1}GB (commitLow={2} physStrikes={3}) -> lms unload --all" -f `
            $m.Phys, $m.Commit, $commitLow, $physStrikes)
        try { & lms unload --all 2>&1 | Out-Null } catch { Log "lms unload failed: $_" }
        Start-Sleep -Seconds 10

        $m = Get-Mem
        if ($m.Commit -lt $MinFreeCommitGB -or $m.Phys -lt $MinFreePhysGB) {
            $fat = Get-Process -Name "LM Studio" -ErrorAction SilentlyContinue |
                Sort-Object WorkingSet64 -Descending | Select-Object -First 1
            if ($fat) {
                Log ("STILL CRITICAL: killing PID {0} (WS {1:N1}GB)" -f $fat.Id, ($fat.WorkingSet64 / 1GB))
                Stop-Process -Id $fat.Id -Force
            } else {
                Log "STILL CRITICAL: no LM Studio process found to kill"
            }
        } else {
            Log ("recovered: freePhys={0:N1}GB freeCommit={1:N1}GB" -f $m.Phys, $m.Commit)
        }
        $physStrikes = 0
    }
    Start-Sleep -Seconds $PollSeconds
}
