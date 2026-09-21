@echo off
REM ===================================================================
REM  One-hour MemConflict scale test (persona-parallel + answer-parallel)
REM  Run from E:\code\github\EMIR2:   run_scale1h.bat
REM
REM  Switches:
REM    resume     continue an interrupted run in the same folder
REM    noscore    run only, no scoring
REM    preflight  only check the container and the model channels
REM    quick      small smoke instead of the one-hour workload
REM    dsjudge    score with DeepSeek instead of OpenRouter's gpt-5-mini
REM    dryrun     print the commands it would run, then stop
REM ===================================================================
setlocal
cd /d "%~dp0"

set RUNDIR=Experiment\runs\scale1h
set CONFIG=Experiment\configs\eval_large.yaml
set JUDGECONFIG=
set JUDGEARG=
set START=11
set PERSONAS=4
set MAXSESS=11
set UNITS=http://172.26.94.12:41135
set RESUME=
set NOSCORE=
set PREONLY=
set DRYRUN=

for %%A in (%*) do (
  if /i "%%A"=="resume"    set RESUME=--resume
  if /i "%%A"=="noscore"   set NOSCORE=1
  if /i "%%A"=="preflight" set PREONLY=1
  if /i "%%A"=="dryrun"    set DRYRUN=1
  if /i "%%A"=="dsjudge" (
    set JUDGECONFIG=1
    set JUDGEARG=--config Experiment\configs\eval_large_dsjudge.yaml
  )
  if /i "%%A"=="quick" (
    set RUNDIR=Experiment\runs\scale1h_quick
    set PERSONAS=1
    set MAXSESS=3
  )
)

REM --- network routing -------------------------------------------------------
REM Do NOT set NO_PROXY here. On this machine Python's getproxies() prefers the
REM environment over the Windows registry, so any NO_PROXY value makes it drop
REM the system proxy (127.0.0.1:7897) and dial OpenRouter directly, where
REM OpenAI's models answer 403 "not available in your region". The GPU server is
REM already dialled directly through the registry's 172.26.* ProxyOverride.
REM Clearing them here also protects against a stale value in your shell.
set "NO_PROXY="
set "no_proxy="
echo proxy: system settings only ^(NO_PROXY cleared^)

REM OpenRouter geo-filters its OpenAI endpoint in bursts of a few seconds to a
REM minute; retry harder than the default before giving up.
set MEMCONFLICT_CHAT_RETRIES=5
set MEMCONFLICT_CHAT_RETRY_BACKOFF=3

echo.
echo ==^> preflight: Ollama container %UNITS%
echo    (loads the model, can take 1-2 minutes when it is cold)
python Experiment\tools\check_ollama_units.py --units %UNITS%
if errorlevel 1 goto :preflight_failed

echo.
echo ==^> preflight: runner channels via %CONFIG%  (memory_builder / answer_model)
python Experiment\tools\check_channels.py --config %CONFIG% --roles memory_builder,answer_model
if errorlevel 1 goto :preflight_failed

echo.
echo ==^> preflight: judge channel (gpt-5-mini via OpenRouter's OpenAI endpoint)
python Experiment\tools\check_channels.py --config %CONFIG% --roles judge_model
if errorlevel 1 (
  echo    WARNING: the judge channel is geo-filtered right now. The run does not need it,
  echo             but scoring will: use  run_scale1h.bat dsjudge  or wait and retry.
)

if defined PREONLY (
  echo.
  echo preflight only: nothing was run.
  goto :eof
)

if not exist "%RUNDIR%" mkdir "%RUNDIR%"

if defined DRYRUN (
  echo.
  echo ==^> dry run: these are the commands this file would execute
  echo python -u Experiment\run_experiment.py --config %CONFIG% --start-index %START% --persona-limit %PERSONAS% --max-sessions %MAXSESS% --persona-workers 4 --answer-workers 4 --extraction-workers 2 --entity-judge-workers 1 --ollama-units %UNITS% --output-dir %RUNDIR% %RESUME%
  echo python -u Experiment\run_scoring.py --run-dir %RUNDIR% %JUDGEARG%
  echo.
  echo nothing was run.
  goto :eof
)

echo.
echo ==^> running the scale test
echo    personas %PERSONAS% x %MAXSESS% sessions from index %START%, unit %UNITS%
echo    tip: append  ^> %RUNDIR%\console.log 2^>^&1  to this command to keep a log
python -u Experiment\run_experiment.py ^
  --config %CONFIG% ^
  --start-index %START% ^
  --persona-limit %PERSONAS% ^
  --max-sessions %MAXSESS% ^
  --persona-workers 4 ^
  --answer-workers 4 ^
  --extraction-workers 2 ^
  --entity-judge-workers 1 ^
  --ollama-units %UNITS% ^
  --output-dir %RUNDIR% %RESUME%
set RUNEXIT=%ERRORLEVEL%

echo.
echo ==^> run summary
if exist "%RUNDIR%\run_meta.json" (
  REM The summary is a Python tool: a printf-style percent sign inside an
  REM inline -c string is expanded by cmd.exe and breaks the command.
  python Experiment\tools\run_summary.py --run-dir "%RUNDIR%"
) else (
  echo    run_meta.json missing: the run did not finish, re-run with:  run_scale1h.bat resume
)

if not defined NOSCORE (
  echo.
  echo ==^> scoring: one judge pass -^> table3 / table5 / table6
  python -u Experiment\run_scoring.py --run-dir %RUNDIR% %JUDGEARG%
  if errorlevel 1 (
    if defined JUDGECONFIG (
      echo    scoring failed with the DeepSeek judge too - check the messages above
    ) else (
      echo.
      echo    OpenRouter judge failed with a geo 403. Retrying with DeepSeek direct.
      echo    NOTE: those tables use deepseek-chat as judge, not the protocol's
      echo          openai/gpt-5-mini - say so when you report the numbers.
      python -u Experiment\run_scoring.py --run-dir %RUNDIR% --config Experiment\configs\eval_large_dsjudge.yaml
      if errorlevel 1 echo    scoring failed with DeepSeek too - check the messages above
    )
  )
)

echo.
echo ==^> artifacts in %RUNDIR%
dir /b "%RUNDIR%"

if not "%RUNEXIT%"=="0" (
  echo.
  echo the run exited with code %RUNEXIT% - see errors.jsonl
  echo continue with:  run_scale1h.bat resume
)
set FINALEXIT=%RUNEXIT%
endlocal & exit /b %FINALEXIT%

:preflight_failed
echo.
echo preflight failed - nothing was run.
echo   * openrouter.ai must NOT be in NO_PROXY; keep the local proxy running.
echo   * if OpenRouter is unusable, add:  dsjudge   (scores with DeepSeek direct)
endlocal
exit /b 1
