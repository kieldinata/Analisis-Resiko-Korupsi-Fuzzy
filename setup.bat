@echo off
echo Memulai instalasi requirement...

:: 1. Memastikan pip terupdate dan install requirement
python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
    echo Instalasi selesai.
) else (
    echo File requirements.txt tidak ditemukan!
)

:: 2. Membuat file run.bat
echo Membuat file run.bat...
(
echo @echo off
echo echo Menjalankan aplikasi...
echo python -m streamlit run app.py
echo pause
) > run.bat

echo.
echo ====================================================
echo Setup selesai!
echo Silakan jalankan file 'run.bat' untuk membuka aplikasi.
echo ====================================================
pause