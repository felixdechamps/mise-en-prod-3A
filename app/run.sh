#!/bin/bash
python src/train.py
uvicorn app.api:app --host 0.0.0.0 --port 8000
