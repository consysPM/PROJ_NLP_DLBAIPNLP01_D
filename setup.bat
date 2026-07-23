@echo off
REM ===========================================================================
REM  Projektbericht DLBAIPNLP01_D - Sentimentanalyse von Produktrezensionen
REM  Einrichtung der virtuellen Umgebung
REM
REM  Legt eine virtuelle Umgebung unter .venv an (falls nicht vorhanden) und
REM  installiert die Abhaengigkeiten aus requirements.txt.
REM
REM  Ausfuehren:  im Projektordner per Doppelklick auf setup.bat
REM               oder in der Eingabeaufforderung:  setup.bat
REM ===========================================================================

setlocal

REM In das Verzeichnis dieser Datei wechseln (wichtig bei Doppelklick).
cd /d "%~dp0"

REM Pruefen, ob Python verfuegbar ist.
python --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python wurde nicht gefunden.
    echo          Bitte Python installieren und zur PATH-Variable hinzufuegen.
    pause
    exit /b 1
)

REM Virtuelle Umgebung anlegen, falls noch nicht vorhanden.
if not exist ".venv\Scripts\activate.bat" (
    echo Erstelle virtuelle Umgebung in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [FEHLER] Konnte die virtuelle Umgebung nicht erstellen.
        pause
        exit /b 1
    )
) else (
    echo Virtuelle Umgebung .venv ist bereits vorhanden.
)

REM Virtuelle Umgebung aktivieren.
call ".venv\Scripts\activate.bat"

REM pip aktualisieren.
echo.
echo Aktualisiere pip ...
python -m pip install --upgrade pip

REM Abhaengigkeiten installieren.
echo.
echo Installiere Abhaengigkeiten aus requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo [FEHLER] Installation der Abhaengigkeiten fehlgeschlagen.
    pause
    exit /b 1
)

echo.
echo ===========================================================================
echo  Fertig. Die virtuelle Umgebung ist eingerichtet.
echo.
echo  Zum spaeteren Aktivieren in einer neuen Eingabeaufforderung:
echo      .venv\Scripts\activate.bat
echo ===========================================================================
pause
