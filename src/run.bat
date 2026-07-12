@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHON=E:\Anaconda\envs\pytorch\python.exe"
set "TRAIN=train_clean.py"
set "DATA_DIR=data\seq_data_librispeech"
set "EXP_BASE=..\exp\full"

if not exist "%PYTHON%" set "PYTHON=python"

if not exist "%TRAIN%" (
    echo ERROR: %TRAIN% not found.
    pause
    exit /b 1
)

for %%S in (0 1 2 3 4) do (
    echo.
    echo ===== Seed %%S =====

    "%PYTHON%" -u "%TRAIN%" ^
        --data-dir "%DATA_DIR%" ^
        --exp-dir "%EXP_BASE%_seed%%S" ^
        --seed %%S ^
        --epochs 100 ^
        --batch-size 25 ^
        --lr 1e-3 ^
        --embed-dim 24 ^
        --depth 3 ^
        --num-heads 1 ^
        --duration-name dur_feat

    if errorlevel 1 goto :failed
)

echo.
echo All runs finished.
pause
exit /b 0

:failed
echo.
echo Training failed.
pause
exit /b 1
