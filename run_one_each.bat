@echo off
setlocal enableextensions

call "%~dp0_env.bat"

%PYTHON% -m pip install --upgrade pip
if exist "scripts\desktop\requirements-desktop.txt" (
  %PYTHON% -m pip install -r "scripts\desktop\requirements-desktop.txt"
)
if exist "scripts\web\requirements-web.txt" (
  %PYTHON% -m pip install -r "scripts\web\requirements-web.txt"
  %PYTHON% -m playwright install chromium
)

REM Run one of each bench with a 1800s window and 64 channels.
REM Update DATA_FILE (and DATA_DIR if needed) to point at a real EDF/NWB file.
set DATA_DIR=data
set DATA_FILE=%DATA_DIR%\synth_64ch_1800s_250hz.edf
set FORMAT=EDF
set TAG=one_each_1800s_64ch

if not exist "%DATA_FILE%" (
  echo ERROR: data file not found: %DATA_FILE%
  echo Please update DATA_FILE in run_one_each.bat to a real EDF/NWB file.
  pause
  exit /b 1
)

echo === IO benches ===
%PYTHON% scripts\io\bench_io_v2.py --tag %TAG% --format %FORMAT% --data-dir %DATA_DIR% --file "%DATA_FILE%" --n-files 1 --runs 1 --window-s 1800 --n-ch 64
%PYTHON% scripts\io\bench_io.py --tag %TAG% --data-dir %DATA_DIR% --n-files 1 --runs 1 --windows 1800 --n-channels 64

echo === Desktop benches ===
%PYTHON% scripts\desktop\bench_fastplotlib.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
%PYTHON% scripts\desktop\bench_mne_rawplot.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
%PYTHON% scripts\desktop\bench_vispy.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
%PYTHON% scripts\desktop\bench_datoviz.py --bench-id TFFR --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
%PYTHON% scripts\desktop\bench_pyqtgraph_tffr_v2.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

%PYTHON% scripts\desktop\bench_vispy_A1_throughput.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
%PYTHON% scripts\desktop\bench_vispy_A2_cadenced.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

%PYTHON% scripts\desktop\bench_pyqtgraph_A1_throughput_v2.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
%PYTHON% scripts\desktop\bench_pyqtgraph_A2_cadenced_v2.py --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

%PYTHON% scripts\desktop\bench_fastplotlib.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
%PYTHON% scripts\desktop\bench_fastplotlib.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

%PYTHON% scripts\desktop\bench_mne_rawplot.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64
%PYTHON% scripts\desktop\bench_mne_rawplot.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64

%PYTHON% scripts\desktop\bench_vispy.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
%PYTHON% scripts\desktop\bench_vispy.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1

%PYTHON% scripts\desktop\bench_datoviz.py --bench-id A1 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1
%PYTHON% scripts\desktop\bench_datoviz.py --bench-id A2 --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --runs 1

echo === Web benches (HTML generation) ===
REM %PYTHON% scripts\web\gen_plotly_html_v2.py --bench-id A1_THROUGHPUT --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --sequence PAN --steps 200 --no-collect
REM %PYTHON% scripts\web\gen_plotly_html_v2.py --bench-id A2_CADENCED --format %FORMAT% --file "%DATA_FILE%" --tag %TAG% --window-s 1800 --n-ch 64 --sequence PAN --steps 200 --target-interval-ms 16.666 --no-collect

endlocal
pause
