<#
.SYNOPSIS
    Throws away the local main.m edit and brings the checkout up to date.

.DESCRIPTION
    Every run machine edits one line of main.m -- `algorithms = {...}`, its share
    of the campaign -- which leaves the file dirty and makes `git pull` refuse.
    That edit is job setup, never work worth keeping, so it is discarded; the
    line is printed first so it can be pasted back after the pull.

    Any OTHER modified tracked file stops the script. Only main.m is disposable,
    and a machine that has real local changes should not have them silently
    reverted or merged over.

    The pull is --ff-only: a run machine has nothing to merge, and a plain pull
    would answer a diverged branch with a merge commit nobody asked for. If it
    refuses, the branch really has diverged and wants a human.

.PARAMETER RepoRoot
    Working tree to update. Defaults to the folder this script sits in.

.PARAMETER Remote
    Remote to fetch from. Defaults to origin.

.EXAMPLE
    .\sync.ps1
    .\sync.ps1 -RepoRoot D:\heuristic
#>
[CmdletBinding()]
param(
    [string] $RepoRoot = $PSScriptRoot,
    [string] $Remote = 'origin'
)

$ErrorActionPreference = 'Stop'

# Native git sets $LASTEXITCODE instead of throwing, so every call goes through
# here; -AllowFail is for the probes whose failure is an answer, not an error.
function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)] [string[]] $Arguments,
        [switch] $AllowFail
    )
    $output = & git -C $RepoRoot $Arguments
    $script:GitExit = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0 -and -not $AllowFail) {
        throw ("git {0} failed (exit {1})" -f ($Arguments -join ' '), $LASTEXITCODE)
    }
    return $output
}

if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }
if (-not (Test-Path -LiteralPath $RepoRoot)) { throw "No such folder: $RepoRoot" }

$RepoRoot = (Invoke-Git @('rev-parse', '--show-toplevel')) | Select-Object -First 1
$branch = (Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD')) | Select-Object -First 1
Write-Output "repo   : $RepoRoot"
Write-Output "branch : $branch"

# --- 1. the working tree -------------------------------------------------- #

# Untracked files are left out: they do not block a fast-forward.
$dirty = @(Invoke-Git @('status', '--porcelain', '--untracked-files=no'))
$others = @($dirty | Where-Object { $_ -and $_.Substring(3).Trim('"') -ne 'main.m' })
if ($others.Count -gt 0) {
    Write-Output ''
    Write-Output 'Modified tracked files other than main.m:'
    $others | ForEach-Object { Write-Output "  $_" }
    throw 'Commit, stash or restore these first -- this script will not touch them.'
}

$mainDirty = @($dirty | Where-Object { $_ -and $_.Substring(3).Trim('"') -eq 'main.m' }).Count -gt 0
$jobLine = $null
if ($mainDirty) {
    # Worth echoing: it is the one thing in the file the machine actually chose.
    $jobLine = (Invoke-Git @('diff', 'HEAD', '--', 'main.m') |
        Where-Object { $_ -match '^\+\s*algorithms\s*=' } | Select-Object -First 1)
    $stat = (Invoke-Git @('diff', 'HEAD', '--shortstat', '--', 'main.m')) | Select-Object -First 1
    Invoke-Git @('restore', '--staged', '--worktree', '--', 'main.m') | Out-Null
    Write-Output "restore: main.m discarded ($($stat.Trim()))"
} else {
    Write-Output 'restore: main.m already clean'
}

# --- 2. fetch and fast-forward -------------------------------------------- #

Invoke-Git @('fetch', $Remote, '--prune') | Out-Null

$upstream = (Invoke-Git @('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}') -AllowFail) |
    Select-Object -First 1
if ($GitExit -ne 0) {
    throw "Branch '$branch' tracks no upstream; nothing to pull. Set one with: git branch --set-upstream-to=$Remote/$branch"
}

$behind = [int]((Invoke-Git @('rev-list', '--count', 'HEAD..@{u}')) | Select-Object -First 1)
$ahead = [int]((Invoke-Git @('rev-list', '--count', '@{u}..HEAD')) | Select-Object -First 1)
Write-Output "fetch  : $upstream -- $behind behind, $ahead ahead"

if ($behind -eq 0) {
    Write-Output 'pull   : already up to date'
} else {
    Invoke-Git @('pull', '--ff-only') | Out-Null
    Write-Output "pull   : fast-forwarded $behind commit(s)"
    Invoke-Git @('log', '--oneline', '-n', [string]$behind) | ForEach-Object { Write-Output "  $_" }
}

$head = (Invoke-Git @('log', '-1', '--pretty=%h %s')) | Select-Object -First 1
Write-Output "head   : $head"

if ($jobLine) {
    Write-Output ''
    Write-Output 'main.m is upstream''s again. The discarded job line was:'
    Write-Output "  $($jobLine.Substring(1).Trim())"
}
