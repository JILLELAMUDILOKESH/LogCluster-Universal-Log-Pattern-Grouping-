# LogCluster — Universal Log Pattern Grouping

A Python-first web app for unsupervised log exploration. LogCluster groups similar log messages, summarizes recurring patterns, and highlights one-off events without requiring labeled data.

## Features

- Friendly Streamlit web interface with paste and file upload options
- CSV, JSON, and JSONL dataset uploads with selectable log-message columns
- Unsupervised clustering with scikit-learn TF-IDF and DBSCAN
- Configurable similarity threshold and minimum pattern size
- Automatic normalization of timestamps, IP addresses, UUIDs, hexadecimal IDs, and numbers
- Searchable pattern table, rare-event view, and CSV export
- Input is analyzed in the local Streamlit app session

## Upload datasets

Choose **Upload a file** and select a `.csv`, `.json`, or `.jsonl` dataset. For structured data, choose the column containing the log messages; LogCluster combines its non-empty rows and discovers patterns. Preview the first few events before analysis, or download the sample CSV to see the expected structure. Plain-text `.log` and `.txt` files continue to work as one event per line.

## Run locally

```powershell
cd LogCluster
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local URL printed by Streamlit. On systems where `py` is not available, replace it with `python`.

## Run tests

```powershell
py -m unittest discover -s tests -v
```

## Project structure

- `app.py` — Streamlit interface, controls, results, and CSV export
- `logcluster.py` — log normalization and unsupervised clustering
- `tests/` — focused tests for normalization, grouping, and input validation
