@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo Distribution-Free Uncertainty Quantification in Machine Learning (R Pipeline)
echo Author: Yuvraj Chandak (Reg. No.: 25BCE2561) ^| Supervisor: Dr. Veeramani S
echo Vellore Institute of Technology, Vellore, Tamil Nadu, India
echo ==============================================================================

:: Find Rscript
where Rscript >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set RSCRIPT_BIN=Rscript
    goto :RUN_SIM
)

if exist "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" (
    set "RSCRIPT_BIN=C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
    goto :RUN_SIM
)

for /d %%D in ("C:\Program Files\R\R-*") do (
    if exist "%%D\bin\Rscript.exe" (
        set "RSCRIPT_BIN=%%D\bin\Rscript.exe"
        goto :RUN_SIM
    )
)

echo [ERROR] Rscript executable could not be found automatically.
echo Please ensure R is installed and added to PATH.
pause
exit /b 1

:RUN_SIM
echo [INFO] Using Rscript at: "!RSCRIPT_BIN!"
echo [INFO] Executing simulation experiments (Monte Carlo M=100 and Boston Housing)...
echo [INFO] Random Seed: 2026

"!RSCRIPT_BIN!" simulation_experiments.R

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Simulation encountered an error.
    pause
    exit /b %ERRORLEVEL%
)

echo [INFO] Running replication-level Student's t-test verification...
"!RSCRIPT_BIN!" update_csv_tables.R

echo ==============================================================================
echo [SUCCESS] Experiments completed successfully!
echo Processed summary tables written to results/ and figures saved to figures/
echo ==============================================================================
pause
