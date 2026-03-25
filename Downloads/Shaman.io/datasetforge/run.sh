#!/bin/bash
echo "Starting The Forge..."
python3 -m pip install -r requirements.txt --quiet
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null
python3 -m streamlit run datasetforge.py
