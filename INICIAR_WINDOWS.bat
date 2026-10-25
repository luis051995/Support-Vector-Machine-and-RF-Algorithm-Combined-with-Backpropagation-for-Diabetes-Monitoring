@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -3.12 -m venv .venv
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
.venv\Scripts\python.exe setup_local.py
if errorlevel 1 goto error
.venv\Scripts\python.exe manage.py migrate --noinput
if errorlevel 1 goto error
.venv\Scripts\python.exe manage.py collectstatic --noinput
if errorlevel 1 goto error
.venv\Scripts\python.exe manage.py create_initial_admin
if errorlevel 1 goto error
echo Abre http://127.0.0.1:8000
.venv\Scripts\python.exe run_local.py
goto end
:error
echo Revisa el error antes de continuar.
pause
:end
