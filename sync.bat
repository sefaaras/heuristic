@echo off
REM ----------------------------------------------------------------------- REM
REM Throws away the local main.m edit and brings the checkout up to date.
REM Double click it, or:  sync.bat [repo folder]
REM
REM Every run machine edits one line of main.m -- `algorithms = {...}`, its
REM share of the campaign -- which leaves the file dirty and makes `git pull`
REM refuse. That edit is job setup, never work worth keeping, so it is thrown
REM away; the line is printed first so it can be pasted back after the pull.
REM
REM Any OTHER modified tracked file stops the run. Only main.m is disposable,
REM and a machine that has real local changes should not have them silently
REM reverted or merged over.
REM
REM The pull is --ff-only: a run machine has nothing to merge, and a plain pull
REM would answer a diverged branch with a merge commit nobody asked for. If it
REM refuses, the branch really has diverged and wants a human.
REM ----------------------------------------------------------------------- REM
setlocal enabledelayedexpansion

set "REMOTE=origin"
set "RC=0"

if "%~1"=="" (cd /d "%~dp0") else (cd /d "%~1")

git rev-parse --show-toplevel >nul 2>nul
if errorlevel 1 (
    echo STOPPED: %CD% is not a git checkout, or git is not on PATH.
    set "RC=1"
    goto :done
)

for /f "usebackq tokens=*" %%R in (`git rev-parse --show-toplevel`) do set "ROOT=%%R"
for /f "usebackq tokens=*" %%B in (`git rev-parse --abbrev-ref HEAD`) do set "BRANCH=%%B"
echo repo   : !ROOT!
echo branch : !BRANCH!

REM --- 1. the working tree ------------------------------------------------ REM

REM -uno leaves untracked files out: they do not block a fast-forward. Column 3
REM onwards is the path, so that is what gets compared -- a rename reads as
REM "old -> new" and lands in OTHERS, which is the cautious side to land on.
set "MAIN_DIRTY="
set "OTHERS="
for /f "usebackq delims=" %%L in (`git status --porcelain -uno`) do (
    set "LINE=%%L"
    set "FILE=!LINE:~3!"
    if /i "!FILE!"=="main.m" (set "MAIN_DIRTY=1") else (set "OTHERS=1")
)

if defined OTHERS (
    echo.
    echo Modified tracked files other than main.m:
    for /f "usebackq delims=" %%L in (`git status --porcelain -uno`) do (
        set "LINE=%%L"
        set "FILE=!LINE:~3!"
        if /i not "!FILE!"=="main.m" echo   !LINE!
    )
    echo.
    echo STOPPED: commit, stash or restore these first -- this script will not
    echo          touch them.
    set "RC=1"
    goto :done
)

if defined MAIN_DIRTY (
    REM Worth echoing: it is the one thing in the file the machine chose itself.
    for /f "usebackq tokens=*" %%J in (`git diff HEAD -- main.m ^| findstr /r /c:"^+ *algorithms"`) do (
        if not defined JOBLINE set "JOBLINE=%%J"
    )
    for /f "usebackq tokens=*" %%S in (`git diff HEAD --shortstat -- main.m`) do set "STAT=%%S"
    git restore --staged --worktree -- main.m
    if errorlevel 1 goto :gitfail
    echo restore: main.m discarded ^(!STAT!^)
) else (
    echo restore: main.m already clean
)

REM --- 2. fetch and fast-forward ------------------------------------------ REM

git fetch %REMOTE% --prune
if errorlevel 1 goto :gitfail

git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >nul 2>nul
if errorlevel 1 (
    echo.
    echo STOPPED: branch !BRANCH! tracks no upstream, so there is nothing to pull.
    echo          set one with:  git branch --set-upstream-to=%REMOTE%/!BRANCH!
    set "RC=1"
    goto :done
)

for /f "usebackq tokens=*" %%U in (`git rev-parse --abbrev-ref --symbolic-full-name "@{u}"`) do set "UPSTREAM=%%U"
for /f "usebackq tokens=*" %%N in (`git rev-list --count HEAD..@{u}`) do set "BEHIND=%%N"
for /f "usebackq tokens=*" %%N in (`git rev-list --count @{u}..HEAD`) do set "AHEAD=%%N"
echo fetch  : !UPSTREAM! -- !BEHIND! behind, !AHEAD! ahead

if "!BEHIND!"=="0" (
    echo pull   : already up to date
) else (
    git pull --ff-only >nul
    if errorlevel 1 goto :gitfail
    echo pull   : fast-forwarded !BEHIND! commit^(s^)
    for /f "usebackq tokens=*" %%C in (`git log --oneline -n !BEHIND!`) do echo   %%C
)

for /f "usebackq tokens=*" %%H in (`git log -1 --oneline`) do set "HEAD_LINE=%%H"
echo head   : !HEAD_LINE!

if defined JOBLINE (
    echo.
    echo main.m is upstream's again. The discarded job line was:
    set "JOBLINE=!JOBLINE:~1!"
    echo   !JOBLINE!
)
goto :done

:gitfail
echo.
echo STOPPED: the git command above failed -- read its message.
set "RC=1"

:done
REM Explorer starts us as `cmd /c "...sync.bat"`, and that console dies the
REM instant we return -- an error message nobody can read. So pause only when
REM our own name is in the command line that started this shell; from an already
REM open prompt it is absent and the script just returns.
REM
REM find.exe by full path: started from a Git Bash shell, cmd inherits a PATH
REM whose /usr/bin comes first, and `find` there is the Unix one, which answers
REM this with an error instead of a match.
echo %cmdcmdline% | "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul
if not errorlevel 1 (
    echo.
    pause
)

exit /b !RC!
