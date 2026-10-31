<# :
@echo off
REM ----------------------------------------------------------------------- REM
REM Progress of the campaign this machine is running: runs finished, time since
REM main.m started the folders, and an estimate of the time left.
REM Double click it, or:  progress.bat [alg ...]
REM
REM With no names it reads main.m's `algorithms = {...}` line, the job line
REM sync.bat talks about. Read-only: it looks at results/<alg>/ and touches
REM nothing, so it is safe to run while main.m is working.
REM
REM How the time left is estimated. parfor hands each worker a chunk of
REM consecutive jobs and runs it from the top index down, so two neighbouring
REM run_info.mat files of one function are written one run apart: their time
REM difference IS a run's duration on this machine. Functions with no such pair
REM yet take a + b * (cost model: D^0.65 per FE plus the recorder), fitted to
REM the measured ones. Time left = the larger of
REM   - the work left divided by the workers busy over the last three hours, and
REM   - the longest unfinished stretch a worker is in the middle of, which is
REM     what decides the end: the last chunks run alone while the rest idle.
REM Replayed on finished campaigns (error as a share of the whole run time):
REM two multi-day ones on 6 workers within -7..+16 % from a fifth of the way
REM on; a 10-hour one on many workers up to 40 % early until two thirds through.
REM Early on it is low by up to half, and flags itself while it is guessing.
REM
REM The script below the batch part is PowerShell; this file is both.
REM ----------------------------------------------------------------------- REM
setlocal
set "PROG_ROOT=%~dp0"
set "PROG_SELF=%~f0"
set "PROG_ARGS=%*"
set "PROG_INTERACTIVE="

REM Started from Explorer the console closes the moment we return, so then the
REM PowerShell part stays open and refreshes on Enter (see sync.bat for why
REM find.exe is called by its full path).
echo %cmdcmdline% | "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul
if not errorlevel 1 set "PROG_INTERACTIVE=1"

powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ([IO.File]::ReadAllText($env:PROG_SELF))"
exit /b %ERRORLEVEL%
#>

$ErrorActionPreference = 'Stop'
$root    = $env:PROG_ROOT.TrimEnd('\')
$results = Join-Path $root 'results'
$window  = 3 * 3600
$runs    = 51

# Mirror of experiment_factory.m and of main.m's all_experiments, in main.m's
# order: functions, dimension, FE budget. Update it with them.
$suites = @(
    @('cec2014_10',  (1..30), 10,  1e5), @('cec2014_30',  (1..30), 30,  3e5),
    @('cec2014_50',  (1..30), 50,  5e5), @('cec2014_100', (1..30), 100, 1e6),
    @('cec2017_10',  (1..30), 10,  1e5), @('cec2017_30',  (1..30), 30,  3e5),
    @('cec2017_50',  (1..30), 50,  5e5), @('cec2017_100', (1..30), 100, 1e6),
    @('cec2020_5',   @(1,2,3,4,5,6,8,9,10), 5, 5e4),
    @('cec2020_10',  (1..10), 10,  1e6), @('cec2020_15',  (1..10), 15,  3e6),
    @('cec2020_20',  (1..10), 20,  1e7),
    @('cec2020rw',   (1..57), 0,   0),
    @('cec2021_10',  (1..10), 10,  2e5), @('cec2021_20',  (1..10), 20,  1e6),
    @('cec2022_10',  (1..12), 10,  2e5), @('cec2022_20',  (1..12), 20,  1e6)
)
# CEC2020RW: Cal_par's dimensions, suite_problem's budget ladder, and the
# measured cost per FE of the problems far above the usual 18-78 us.
$rwD  = @(9,11,7,6,9,38,48,2,3,3,7,7,5,10,7,14,3,4,4,2,5,9,5,7,4,22,10,10,4,3,4,5,
          30,118,153,158,126,126,126,76,74,86,86,30,25,25,25,30,30,30,59,59,59,59,64,64,64)
$rwUs = @{ 1 = 106; 24 = 798; 27 = 111; 33 = 272; 44 = 962 }

# Seconds of one run on one core: search + evaluation, plus the recorder
function Get-RunCost([double]$D, [double]$fe, [double]$usPerFe) {
    $usPerFe * 1e-6 * $fe + 0.19 * [math]::Pow($D / 10, 0.62)
}
function Get-RwBudget([double]$D) {
    if ($D -le 10) { 1e5 } elseif ($D -le 30) { 2e5 } elseif ($D -le 50) { 4e5 }
    elseif ($D -le 150) { 8e5 } else { 1e6 }
}
function Get-Median($list) {
    $a = $list.ToArray()
    [Array]::Sort($a)
    $n = $a.Count
    if ($n % 2) { return $a[($n - 1) / 2] }
    ($a[$n / 2 - 1] + $a[$n / 2]) / 2
}
function Format-Span([double]$seconds) {
    if ($seconds -lt 60) { return '< 1 min' }
    $t = [TimeSpan]::FromSeconds($seconds)
    '{0}d {1:00}:{2:00}' -f [int][math]::Floor($t.TotalDays), $t.Hours, $t.Minutes
}

# Per experiment: function -> position, and the cost model of each function
$expNames = @(); $funcPos = @(); $funcModel = @(); $expModel = @()
foreach ($s in $suites) {
    $pos = @{}; $mod = New-Object double[] $s[1].Count; $i = 0
    foreach ($f in $s[1]) {
        $pos[$f] = $i
        if ($s[0] -eq 'cec2020rw') {
            $D = $rwD[$f - 1]; $us = 40
            if ($rwUs.ContainsKey($f)) { $us = $rwUs[$f] }
            $mod[$i] = Get-RunCost $D (Get-RwBudget $D) $us
        } else {
            $mod[$i] = Get-RunCost $s[2] $s[3] (4.4 * [math]::Pow($s[2] / 10, 0.65))
        }
        $i++
    }
    $expNames += $s[0]; $funcPos += ,$pos; $funcModel += ,$mod
    $expModel += Get-Median ([Collections.Generic.List[double]]$mod)
}
$nExp = $expNames.Count

function Show-Progress {
    $now = Get-Date
    if ($env:PROG_AS_OF) {   # replay: the state as it was at that time
        $now = [datetime]::ParseExact($env:PROG_AS_OF, 'yyyy-MM-dd HH:mm', $null)
    }
    $nowS = $now.ToUniversalTime().Ticks / 1e7

    $mainText = ''
    $mainFile = Join-Path $root 'main.m'
    if (Test-Path -LiteralPath $mainFile) { $mainText = [IO.File]::ReadAllText($mainFile) }
    $pipelineId = [regex]::Match($mainText, "(?m)^\s*pipeline_id\s*=\s*'([^']+)'").Groups[1].Value

    $algs = @()
    if ($env:PROG_ARGS) { $algs = @($env:PROG_ARGS.Trim() -split '\s+' | Where-Object { $_ }) }
    if ($algs.Count -eq 0) {
        $line = [regex]::Match($mainText, '(?m)^\s*algorithms\s*=\s*\{([^}]*)\}').Groups[1].Value
        $algs = @([regex]::Matches($line, "'([^']+)'") | ForEach-Object { $_.Groups[1].Value })
    }
    Write-Host ('progress : {0}   ({1:yyyy-MM-dd HH:mm})' -f $results, $now)
    if ($algs.Count -eq 0) {
        Write-Host ''
        Write-Host "main.m's algorithms line is empty, so there is nothing to follow."
        Write-Host 'Name the algorithms instead:  progress.bat alg1 alg2'
        return
    }
    $nAlg = $algs.Count

    # The job grid in main.m's order (experiment, algorithm, function, run); a block
    # is one function's 51 runs. gT = completion time in seconds, NaN while missing.
    $blockOf = @(); $nb = 0
    for ($e = 0; $e -lt $nExp; $e++) { $blockOf += $nb; $nb += $nAlg * $funcModel[$e].Count }
    $nJobs  = $nb * $runs
    $gT = New-Object double[] $nJobs
    for ($n = 0; $n -lt $nJobs; $n++) { $gT[$n] = [double]::NaN }
    $bE = New-Object int[] $nb; $bA = New-Object int[] $nb; $bF = New-Object int[] $nb
    for ($e = 0; $e -lt $nExp; $e++) {
        $k = $funcModel[$e].Count
        for ($a = 0; $a -lt $nAlg; $a++) {
            for ($i = 0; $i -lt $k; $i++) {
                $b = $blockOf[$e] + $a * $k + $i
                $bE[$b] = $e; $bA[$b] = $a; $bF[$b] = $i
            }
        }
    }

    $starts = @()
    for ($a = 0; $a -lt $nAlg; $a++) {
        $algDir = Join-Path $results $algs[$a]
        $start = $null
        # main.m's stamp, <id>_<yyyy-MM-dd_HH-mm-ss>_<commit>: when the campaign first started
        if ($pipelineId -and (Test-Path -LiteralPath $algDir)) {
            foreach ($s in [IO.Directory]::GetFiles($algDir, "${pipelineId}_*")) {
                $m = [regex]::Match([IO.Path]::GetFileName($s), '_(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)_')
                if ($m.Success) {
                    $t = [datetime]::ParseExact($m.Groups[1].Value, 'yyyy-MM-dd_HH-mm-ss', $null)
                    if ($null -eq $start -or $t -lt $start) { $start = $t }
                }
            }
        }
        $starts += $start

        for ($e = 0; $e -lt $nExp; $e++) {
            $expDir = Join-Path $algDir $expNames[$e]
            if (-not (Test-Path -LiteralPath $expDir)) { continue }
            $pos = $funcPos[$e]
            $base = ($blockOf[$e] + $a * $funcModel[$e].Count) * $runs
            $files = ([IO.DirectoryInfo]$expDir).EnumerateFiles('run_info.mat', [IO.SearchOption]::AllDirectories)
            foreach ($fi in $files) {
                # main.m's own completion test: a 0-byte run_info is a torn write
                if ($fi.Length -le 0) { continue }
                $t = $fi.LastWriteTimeUtc.Ticks / 1e7
                if ($t -gt $nowS) { continue }
                $runOk = $fi.Directory.Name -match '^run(\d+)$' -and [int]$Matches[1] -le $runs
                if (-not $runOk) { continue }
                $r = [int]$Matches[1]
                if ($fi.Directory.Parent.Name -notmatch '^F(\d+)$') { continue }
                $f = [int]$Matches[1]
                if (-not $pos.ContainsKey($f)) { continue }
                $gT[$base + $pos[$f] * $runs + $r - 1] = $t
            }
        }
    }

    # A chunk runs from its top index down, so run r finished one run after run r+1
    $sampF = @{}; $sampE = @{}
    for ($b = 0; $b -lt $nb; $b++) {
        $n0 = $b * $runs
        for ($r = 0; $r -lt $runs - 1; $r++) {
            $dt = $gT[$n0 + $r] - $gT[$n0 + $r + 1]
            if (-not ($dt -ge 0 -and $dt -lt 21600)) { continue }   # NaN fails too
            if (-not $sampF.ContainsKey($b)) { $sampF[$b] = New-Object Collections.Generic.List[double] }
            $sampF[$b].Add($dt)
            $ke = $bA[$b] * $nExp + $bE[$b]
            if (-not $sampE.ContainsKey($ke)) { $sampE[$ke] = New-Object Collections.Generic.List[double] }
            $sampE[$ke].Add($dt)
        }
    }
    $medE = @{}
    foreach ($ke in $sampE.Keys) { $medE[$ke] = Get-Median $sampE[$ke] }

    # Unmeasured experiments: duration = fa + fb * model, fitted per algorithm (pooled as fallback)
    $fa = New-Object double[] ($nAlg + 1); $fb = New-Object double[] ($nAlg + 1)
    for ($a = 0; $a -le $nAlg; $a++) {
        $xs = New-Object Collections.Generic.List[double]; $ys = New-Object Collections.Generic.List[double]
        foreach ($ke in $medE.Keys) {
            if ($a -lt $nAlg -and [math]::Floor($ke / $nExp) -ne $a) { continue }
            $xs.Add($expModel[$ke % $nExp]); $ys.Add($medE[$ke])
        }
        $fa[$a] = [double]::NaN; $fb[$a] = [double]::NaN
        if ($xs.Count -eq 0) { continue }
        $ratio = New-Object Collections.Generic.List[double]
        for ($i = 0; $i -lt $xs.Count; $i++) { $ratio.Add($ys[$i] / $xs[$i]) }
        $fa[$a] = 0; $fb[$a] = Get-Median $ratio
        $mx = ($xs | Measure-Object -Average).Average; $my = ($ys | Measure-Object -Average).Average
        $sxx = 0.0; $sxy = 0.0
        for ($i = 0; $i -lt $xs.Count; $i++) {
            $sxx += ($xs[$i] - $mx) * ($xs[$i] - $mx); $sxy += ($xs[$i] - $mx) * ($ys[$i] - $my)
        }
        if ($sxx -gt 0 -and $sxy -gt 0) { $fb[$a] = $sxy / $sxx; $fa[$a] = [math]::Max($my - $fb[$a] * $mx, 0) }
    }
    # Nothing timed yet: the bare model still weights the work done, but gives no time
    $noTiming = [double]::IsNaN($fb[$nAlg])
    if ($noTiming) { $fa[$nAlg] = 0; $fb[$nAlg] = 1 }

    $bD = New-Object double[] $nb
    $unmeasured = @{}
    for ($b = 0; $b -lt $nb; $b++) {
        $e = $bE[$b]; $a = $bA[$b]; $ke = $a * $nExp + $e
        $mod = $funcModel[$e][$bF[$b]]
        if ($sampF.ContainsKey($b) -and $sampF[$b].Count -ge 3) { $bD[$b] = Get-Median $sampF[$b] }
        elseif ($medE.ContainsKey($ke)) { $bD[$b] = $medE[$ke] * $mod / $expModel[$e] }
        else {
            $unmeasured[$e] = 1
            $fit = $a
            if ([double]::IsNaN($fb[$a])) { $fit = $nAlg }
            $bD[$b] = $fa[$fit] + $fb[$fit] * $mod
        }
    }

    # Totals, the busy workers of the window, and the unfinished stretch each worker is in
    $doneN = 0; $doneW = 0.0; $totalW = 0.0; $busyW = 0.0; $tFirst = [double]::MaxValue; $tLast = 0.0
    $algDone = New-Object int[] $nAlg; $expDone = New-Object int[] $nExp; $expTotal = New-Object int[] $nExp
    for ($b = 0; $b -lt $nb; $b++) {
        $d = $bD[$b]; $n0 = $b * $runs
        $expTotal[$bE[$b]] += $runs
        for ($r = 0; $r -lt $runs; $r++) {
            $totalW += $d
            $t = $gT[$n0 + $r]
            if ([double]::IsNaN($t)) { continue }
            $doneN++; $doneW += $d; $algDone[$bA[$b]]++; $expDone[$bE[$b]]++
            if ($t -lt $tFirst) { $tFirst = $t }
            if ($t -gt $tLast) { $tLast = $t }
        }
    }
    $w0 = [math]::Max($nowS - $window, $tFirst)
    $stretch = 0.0; $open = 0.0
    for ($n = 0; $n -lt $nJobs; $n++) {
        $d = $bD[[math]::Floor($n / $runs)]
        $t = $gT[$n]
        if ([double]::IsNaN($t)) { $open += $d; continue }
        if ($t -gt $w0) { $busyW += $d }
        # the job above a missing stretch finished lately: a worker is coming down it
        if ($open -gt 0 -and $nowS - $t -lt [math]::Max(3600, 3 * $d)) { $stretch = [math]::Max($stretch, $open) }
        $open = 0.0
    }

    Write-Host ''
    Write-Host ('{0,-16} {1,15} {2,8}   {3,-16}   {4}' -f 'algorithm', 'runs done', '', 'started', 'elapsed')
    for ($a = 0; $a -lt $nAlg; $a++) {
        $began = 'not started'; $elapsed = ''
        if ($null -ne $starts[$a]) {
            $began = $starts[$a].ToString('yyyy-MM-dd HH:mm')
            $elapsed = Format-Span ($now - $starts[$a]).TotalSeconds
        }
        $tot = $nb * $runs / $nAlg
        Write-Host ('{0,-16} {1,6} / {2,-6} {3,6:0.0} %   {4,-16}   {5}' -f `
                    $algs[$a], $algDone[$a], $tot, (100.0 * $algDone[$a] / $tot), $began, $elapsed)
    }

    Write-Host ''
    Write-Host ('{0,-16} {1,15}' -f 'experiment', 'runs done')
    for ($e = 0; $e -lt $nExp; $e++) {
        $pct = 100.0 * $expDone[$e] / $expTotal[$e]
        $bar = ('#' * [int][math]::Floor($pct / 5)).PadRight(20, '.')
        Write-Host ('{0,-16} {1,6} / {2,-6} {3,6:0.0} %   {4}' -f $expNames[$e], $expDone[$e], $expTotal[$e], $pct, $bar)
    }

    Write-Host ''
    Write-Host ('total            {0} / {1} runs ({2:0.0} %), about {3:0} % of the work' -f `
                $doneN, $nJobs, (100.0 * $doneN / $nJobs), (100.0 * $doneW / $totalW))
    $start = $starts | Where-Object { $null -ne $_ } | Sort-Object | Select-Object -First 1
    if ($null -ne $start) {
        Write-Host ('elapsed          {0}   (since {1:yyyy-MM-dd HH:mm})' -f (Format-Span ($now - $start).TotalSeconds), $start)
    }

    $matlab = @(Get-Process -Name MATLAB -ErrorAction SilentlyContinue).Count
    $state = 'not running'
    if ($matlab -gt 0) { $state = "running, $matlab processes" }
    if ($env:PROG_AS_OF) { $state = 'not checked (replay)' }
    if ($doneN -gt 0) {
        Write-Host ('last result      {0} ago   (MATLAB: {1})' -f (Format-Span ($nowS - $tLast)), $state)
    } else {
        Write-Host ('last result      none yet   (MATLAB: {0})' -f $state)
    }

    if ($doneN -ge $nJobs) {
        Write-Host ''
        Write-Host 'All runs are on disk.'
        return
    }
    if ($busyW -le 0 -or $nowS - $w0 -lt 600 -or $noTiming) {
        Write-Host 'time left        not enough results in the last 3 h to estimate it'
        return
    }
    $busy = $busyW / ($nowS - $w0)
    $left = [math]::Max(($totalW - $doneW) / $busy, $stretch)
    Write-Host ('busy workers     {0:0.0}   (over the last 3 h)' -f $busy)
    Write-Host ('time left        ~ {0}   finishes ~ {1:yyyy-MM-dd HH:mm}' -f (Format-Span $left), $now.AddSeconds($left))
    if ($unmeasured.Count -gt 0) {
        Write-Host ('                 rough: {0} of {1} experiments have no timed run yet, so it is likely low' -f `
                    $unmeasured.Count, $nExp)
    }
    if ($matlab -eq 0 -and -not $env:PROG_AS_OF) {
        Write-Host ''
        Write-Host 'MATLAB is not running: the campaign is stopped, the estimate assumes it resumes now.'
    }
}

while ($true) {
    Show-Progress
    # `cmd /c progress.bat` from a script looks like a double click; its stdin does not
    if ($env:PROG_INTERACTIVE -ne '1' -or [Console]::IsInputRedirected) { break }
    Write-Host ''
    $key = Read-Host 'Enter = refresh, q = quit'
    if ($null -eq $key -or $key -eq 'q') { break }
    Clear-Host
}
