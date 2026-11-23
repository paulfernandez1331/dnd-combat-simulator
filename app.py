"""Streamlit entry point. Run with: streamlit run app.py"""

from dotenv import load_dotenv

from src.paths import ROOT
from src.ui import main

load_dotenv(ROOT / ".env")
main()
