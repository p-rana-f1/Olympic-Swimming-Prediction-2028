OLYMPIC SWIMMING PROJECT - WINDOWS QUICK START

1. Install Python 3.11 or newer and Visual Studio Code.
2. Open this folder in VS Code (File > Open Folder).
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   py -m venv .venv
5. Activate it in Command Prompt terminal:
   .venv\Scripts\activate.bat
   (In PowerShell, use: .\.venv\Scripts\Activate.ps1)
6. Install dependencies:
   python -m pip install -r requirements.txt
7. Start website:
   python -m streamlit run app.py
8. Open http://localhost:8501 if it does not open automatically.
9. To stop: press Ctrl+C in the terminal.

The app includes individual events only and needs at least 5 years
of results per country/event to produce a forecast.
