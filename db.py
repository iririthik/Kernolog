#!/usr/bin/env python3
"""
Live Log Embedding System (journalctl -> FAISS)

Features:
 - Streams `journalctl -f`
 - Deduplicates repeated messages by normalizing lines (strips timestamps, PIDs)
 - Summarizes frequent repeats every 10s as “⏱ … repeated Nx”
 - Embeds logs in real time with SentenceTransformer + FAISS
 - Search with adjustable k and display mode
"""

import threading
import subprocess
import queue
import time
import re
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer

# --- Configuration ---
JOURNALCTL_CMD = ["journalctl", "-f", "-o", "short"]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 16
DEFAULT_K = 5
FLUSH_INTERVAL = 10  # seconds between repeated-log summaries

# --- Shared objects ---
log_queue = queue.Queue()
metadata = []  # list of {"id": int, "text": str, "timestamp": float}
index = faiss.IndexFlatL2(EMBED_DIM)
model = SentenceTransformer(EMBED_MODEL_NAME)

# --- Repeat cache ---
repeat_cache = {}  # {normalized_message: count}
cache_lock = threading.Lock()


# --- Utility: normalize logs ---
def normalize_log(line: str) -> str:
    """
    Strip volatile fields (timestamps, PIDs, hostnames) for repeat detection.
    Example:
      "Nov 04 23:58:33 archlinux systemd[1]: ollama.service failed"
      -> "systemd: ollama.service failed"
    """
    # Remove leading timestamp and hostname
    line = re.sub(r'^[A-Z][a-z]{2}\s+\d+\s+\d+:\d+:\d+\s+\S+\s+', '', line)
    # Remove PID markers like [1234]
    line = re.sub(r'\[\d+\]', '', line)
    # Normalize multiple spaces
    line = re.sub(r'\s+', ' ', line)
    return line.strip()


# --- Thread: Watch journalctl ---
def watch_journalctl():
    proc = subprocess.Popen(JOURNALCTL_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        line = proc.stdout.readline()
        if not line:
            continue
        line = line.rstrip("\n")
        if not line:
            continue

        normalized = normalize_log(line)
        with cache_lock:
            repeat_cache[normalized] = repeat_cache.get(normalized, 0) + 1


# --- Thread: Periodically flush repeated logs ---
def repeat_flusher():
    next_id = 0
    while True:
        time.sleep(FLUSH_INTERVAL)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with cache_lock:
            items = list(repeat_cache.items())
            repeat_cache.clear()

        for msg, count in items:
            ts = time.time()
            if count == 1:
                # Single occurrence, push directly
                log_queue.put((next_id, msg, ts))
            else:
                # Summarize multiple repeats
                summary = f"⏱ {now} | \"{msg}\" repeated {count}x"
                log_queue.put((next_id, summary, ts))
            next_id += 1


# --- Thread: Embed & index logs ---
def embed_worker():
    batch_ids = []
    batch_texts = []
    batch_timestamps = []

    while True:
        try:
            _id, text, ts = log_queue.get(timeout=1.0)
        except queue.Empty:
            # flush remaining batch if any
            if batch_texts:
                embeddings = model.encode(batch_texts, convert_to_numpy=True)
                index.add(embeddings)
                for i, txt, tstamp in zip(batch_ids, batch_texts, batch_timestamps):
                    metadata.append({"id": i, "text": txt, "timestamp": tstamp})
                batch_ids.clear()
                batch_texts.clear()
                batch_timestamps.clear()
            continue

        batch_ids.append(_id)
        batch_texts.append(text)
        batch_timestamps.append(ts)

        if len(batch_texts) >= BATCH_SIZE:
            embeddings = model.encode(batch_texts, convert_to_numpy=True)
            index.add(embeddings)
            for i, txt, tstamp in zip(batch_ids, batch_texts, batch_timestamps):
                metadata.append({"id": i, "text": txt, "timestamp": tstamp})
            batch_ids.clear()
            batch_texts.clear()
            batch_timestamps.clear()


# --- Query function ---
def search_query(q: str, k: int, display_mode: str):
    q_emb = model.encode([q], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(metadata):
            meta = metadata[idx]
            if display_mode == "raw":
                results.append(meta["text"])
            else:
                results.append(f"{meta['timestamp']:.3f} | dist={dist:.3f} | {meta['text']}")
    return results


# --- Start background threads ---
watcher_thread = threading.Thread(target=watch_journalctl, daemon=True)
flusher_thread = threading.Thread(target=repeat_flusher, daemon=True)
embed_thread = threading.Thread(target=embed_worker, daemon=True)

watcher_thread.start()
flusher_thread.start()
embed_thread.start()


# --- Main interaction loop ---
if __name__ == "__main__":
    print("🧠 Live log embedding system started (journalctl + deduplication).")
    time.sleep(5)

    while True:
        try:
            line = input("Enter search query (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        # Parse search options
        k = DEFAULT_K
        display_mode = "pretty"
        parts = line.split()
        filtered_parts = []
        for p in parts:
            if p.startswith("k="):
                try:
                    k = int(p.split("=", 1)[1])
                except ValueError:
                    print("Invalid k value; using default.")
            elif p.startswith("display="):
                mode = p.split("=", 1)[1]
                if mode in ("raw", "pretty"):
                    display_mode = mode
                else:
                    print("Invalid display mode; using pretty.")
            else:
                filtered_parts.append(p)

        query_text = " ".join(filtered_parts)
        if not query_text:
            print("Empty query text; please provide a search term.")
            continue

        hits = search_query(query_text, k, display_mode)
        print("-" * 80)
        print(f"Top {k} results (display={display_mode}):")
        for r in hits:
            print(r)
        print("-" * 80)
