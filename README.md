# Kernolog

**Real-time Linux log monitoring and semantic search **

Kernolog streams live system logs from `journalctl`, deduplicates repeated entries using the Drain3 log template miner, embeds them with a SentenceTransformer model, and stores them in a hybrid SQLite + binary vector store for fast semantic similarity search.

---

## Features

- **Live log ingestion** via `journalctl` in JSON mode, capturing message, priority, and unit
- **Log normalization & deduplication** using [Drain3](https://github.com/logpai/Drain3) — repeated log lines are collapsed into templates (e.g. `"User <*> logged in"`)
- **Parameter extraction** — variables like usernames, device names, and IPs are stored separately and highlighted in results
- **AI embeddings** using `all-MiniLM-L6-v2` (via `sentence-transformers`)
- **Three-tier classification** — logs are bucketed into `error` (priority ≤ 3), `warning` (priority 4), and `debug` (priority ≥ 5)
- **Two-phase semantic search** — a fast broad pass over template vectors, followed by live re-ranking with hydrated (parameter-restored) sentences
- **Recency-biased search** — use keywords like `now`, `latest`, or `recent` to surface the most recent relevant logs
- **Desktop alerts** via `notify-send` for critical errors
- **Interactive shell** for querying logs in real time

---

## Architecture

```
journalctl (JSON)
      │
      ▼
 [Collector]       collector/core.py
      │  raw log dicts
      ▼
 [Normalizer]      normalizer/core.py
      │  template + params + priority
      ▼
 [Engine]          engine.py
      │  batched embedding + storage
      ▼
 [Storage]         storage.py
   ├── {category}.sqlite   (templates, occurrences, parameters)
   └── {category}.bin      (float32 embedding vectors)
      │
      ▼
 [Shell]           shell.py
   └── semantic search CLI
```

---

## Requirements

- Python 3.8+
- Linux with `systemd` (for `journalctl`)
- `notify-send` (optional, for desktop alerts on critical errors)

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/iririthik/Kernolog.git
cd Kernolog
```

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install sentence-transformers drain3 colorama numpy
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~90 MB) from HuggingFace automatically.

### 4. (Optional) Install desktop notifications

```bash
# Debian/Ubuntu
sudo apt install libnotify-bin

# Arch
sudo apt install libnotify
```

### 5. Verify journalctl access

Kernolog reads from `journalctl`. If you get a permissions error, add your user to the `systemd-journal` group:

```bash
sudo usermod -aG systemd-journal $USER
# Log out and back in for this to take effect
```

---

## Quick Start

### Run everything (recommended)

```bash
python boot.py
```

This starts the engine in the background and opens the search shell in a new terminal window. Indexed data is written to `./gen_data/`.

### Or run components separately

**Terminal 1 — start the engine:**
```bash
python engine.py
```

**Terminal 2 — open the search shell:**
```bash
python shell.py
```

---

## Search Shell Usage

```
Kernolog> search <category> <query>
```

**Categories:** `error`, `warning`, `debug`

| Example command | What it does |
|---|---|
| `search error disk failure` | Semantic search for disk-related errors |
| `search warning latest` | Most recent warnings |
| `search error usb device now` | Recent USB errors, time-prioritized |
| `search debug network` | Debug logs related to networking |

**Tips:**
- Use `now`, `latest`, `recent`, `last`, `today`, or `current` to sort results by time instead of relevance score
- Type `clear` to clear the screen
- Type `exit` or `quit` to close the shell

**Example output:**
```
--- ERROR Results ---
[Score:0.87] 14:23:01.042 | kernel: usb disconnect, device number 3
[Score:0.74] 14:19:55.118 | kernel: EXT4-fs error on sda1
--------------------------------------------------
```

Matched parameters are highlighted in yellow.

---

## Data Storage

All indexed data lives in `./gen_data/`:

| File | Contents |
|---|---|
| `error.sqlite` | Templates, occurrences, and extracted parameters for errors |
| `error.bin` | Raw float32 embedding vectors (384-dim) |
| `warning.sqlite` / `warning.bin` | Same for warnings |
| `debug.sqlite` / `debug.bin` | Same for debug/info logs |

---

## Using as a Library

**Watch live logs with a custom callback:**

```python
from collector import watch

def my_handler(log):
    print(log['priority'], log['message'])

watch(my_handler)
```

**Run the full normalized pipeline:**

```python
from normalizer import process_stream

def my_handler(log_line):
    print(log_line)

process_stream(my_handler)
```

---

## License

[AGPL-3.0](LICENSE) — if you run a modified version as a network service, you must make your source code available.

---

## Contributing

Contributions are welcome. See the repository's contributing guidelines.

## Acknowledgements

Built with [Drain3](https://github.com/logpai/Drain3), [sentence-transformers](https://www.sbert.net/), and [NumPy](https://numpy.org/).
