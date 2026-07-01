# HUMEMBR

This repository contains the implementation of **HUMEMBR: Learning Human Routines for Predictive Embodied Navigation**.

- **Project website:** https://samirahuber.github.io/humembr/
- **arXiv:** https://arxiv.org/abs/2606.30404
- **Paper (PDF):** https://arxiv.org/pdf/2606.30404

HUMEMBR enables a Boston Dynamics Spot robot to build and query a long-term, human-centric semantic memory of its environment, supporting natural-language tasks such as "find person X" or "who was in the kitchen this morning?".

If you use this code in your research, please cite the HUMEMBR paper:

```bibtex
@inproceedings{huber2026humembr,
  title     = {HUMEMBR: Learning Human Routines for Predictive Embodied Navigation},
  author    = {Huber, Samira and Pelzer, Klaas and Nguyen, Duc M. and Xiao, Xuesu and Pirk, S{\"o}ren},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
}
```

---

## Quick-start checklist

To get the system running, follow these steps in order:

1. [Install uv and Python](#1-install-uv-and-python)
2. [Clone and install dependencies](#2-clone-and-install-dependencies)
3. [Configure the environment file](#3-configure-the-environment-file)
4. [Configure `config.toml`](#4-configure-configtoml)
5. [Start the database](#5-start-the-database)
6. [Run database migrations](#6-run-database-migrations)
7. [Record or download a GraphNav map](#7-record-or-download-a-graphnav-map)
8. [Download KPR weights](#8-download-kpr-weights)
9. [Start a captioning model](#9-start-a-captioning-model)
10. [Start the application](#10-start-the-application)

---

## 1. Install uv and Python

This project uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

```bash
# Install uv (if you don't have it yet)
# https://docs.astral.sh/uv/getting-started/installation/

# Install the required Python version
uv python install 3.10.3
```

**The code was tested on macOS and Ubuntu only.**

---

## 2. Clone and install dependencies

```bash
git clone <repository-url>
cd humembr
uv sync
```

All Python dependencies should now be installed.

---

## 3. Configure the environment file

Create `src/humembr/.env` (or copy from `src/humembr/example.env`) and set at least:

```bash
HUMEMBR_CONFIG_PATH="/absolute/path/to/src/humembr/config.toml"
DATABASE_URL="postgresql://postgres:mysecretpassword@localhost:5434/spot-db?sslmode=disable&connect_timeout=5"
```

You will also need an API key for the captioning/LLM backend you choose (e.g., a Gemini key if you use the Gemini-based components). Set it via `.bashrc`, `.zshrc`, or another mechanism your deployment expects.

---

## 4. Configure `config.toml`

Open `src/humembr/config.toml` and set the **absolute** paths and endpoints:

| Key | What to set | Example |
|-----|-------------|---------|
| `map_dir` | Absolute path to your GraphNav map directory | `/home/user/humembr/maps/downloaded_graph` |
| `img_dir` | Absolute path for image storage | `/home/user/humembr/testing-images` |
| `backup_dir` | Absolute path for backups | `/home/user/humembr/backups` |
| `hostname` | Robot IP address | `<robot-ip>` |
| `caption_url` | URL of your captioning service | `http://localhost:8000/v1` or `http://localhost:11434/v1` |
| `caption_model` | Model name | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` |

For a full list of options and defaults, see `src/humembr/util/config.py`.

---

## 5. Start the database

Start a PostgreSQL instance with the `pgvector` extension:

```bash
docker run -d \
  --name my-pgvector-db \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -v pgvector_data:/var/lib/postgresql \
  -p 5434:5432 \
  pgvector/pgvector:pg18-trixie
```

To restart an existing container:

```bash
docker container start my-pgvector-db
```

---

## 6. Run database migrations

Install `dbmate` and run the migrations from `src/humembr`:

```bash
npm install -g dbmate
cd src/humembr
dbmate up
```

---

## 7. Record or download a GraphNav map

HUMEMBR relies on Boston Dynamics' GraphNav service. You have two options:

### Option A: Record your own map

Follow the official Spot SDK example to record a map of your environment:
https://dev.bostondynamics.com/python/examples/graph_nav_command_line/readme#recording-service-command-line

After recording, download the map and set `map_dir` in `src/humembr/config.toml` to the downloaded directory.

### Option B: Use a pre-recorded map

If you have a released map for the paper, download it and point `map_dir` to the extracted folder.

---

## 8. Download KPR weights

Download the keypoint-reidentification weights from:
https://drive.google.com/file/d/1Ah4iOjz_VOMnl0QUKYaRb5v5AOfLyG51/view?usp=sharing

Extract/copy them to:

```
src/humembr/processing/pretrained
```

---

## 9. Start a captioning model

Choose **one** of the following backends and start it before launching HUMEMBR.

### vLLM (recommended, best performance)

Start a vLLM server with a vision-language model, for example:

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data \
  --enable-auto-tool-choice \
  --enable-expert-parallel \
  --tool-call-parser hermes \
  --async-scheduling
```

Then set in `config.toml`:

```toml
caption_url = "http://localhost:8000/v1"
caption_model = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
```

The client code is `src/humembr/processing/qwen_openai.py`.

### Ollama

```bash
ollama serve
ollama pull qwen2.5vl:7b
```

Then set in `config.toml`:

```toml
caption_url = "http://localhost:11434/v1"
caption_model = "qwen2.5vl:7b"
```

The client code is `src/humembr/processing/qwen.py`.

### Other OpenAI-compatible services

Use `src/humembr/processing/qwen_openai.py` and set the URL and model name in `config.toml`.

---

## 10. Start the application

Start the three required components in separate terminals:

```bash
# Terminal 1: robot control interface
uv run -m humembr.robot.main

# Terminal 2: web chat interface and agent
uv run -m humembr.server.app

# Terminal 3: image captioning (pick one)
uv run -m humembr.processing.qwen          # Ollama backend
# or
uv run -m humembr.processing.qwen_openai   # vLLM / OpenAI-compatible backend
```

Open `http://127.0.0.1:5050/` in your browser.

If you **only** want to record data and not let the agent control the robot, set in `config.toml`:

```toml
enable_ctrl = false
```

---

## Next steps

- Drive the robot with prompts like `"go to waypoint 42"` to collect observations.
- After enough people have been clustered, name them in the UI.
- Then query the memory with prompts like `"find person-1"` or `"who was in the kitchen this morning?"`.

---

## Backup, restore, and dataset ingestion

### Backup the current image log

```bash
uv run -m scripts.backup_image_queue backup
```

### Restore a backup or dataset

```bash
uv run -m humembr.scripts.backup_image_queue restore --archive_path 'path/to/backup.tar.gz'
```

This imports image observations but **not** person embeddings. Generate embeddings afterwards:

```bash
uv run -m scripts.insert_missing_reid
```

### Load the released dataset

> **The released dataset is currently private and not available for public download.**
> If you have access to the archive, follow the steps below.

```bash
uv run -m humembr.scripts.backup_image_queue restore --archive_path 'path/to/downloaded/dataset.tar.gz'
uv run -m scripts.insert_missing_reid
```

---

## Delete recent data

```bash
docker exec -it my-pgvector-db psql -U postgres
```

```sql
\c spot-db

DELETE FROM image_queue
WHERE creation_timestamp::date = CURRENT_DATE;
```

---

## Important files

If you want to modify the LLM behavior or captioning pipeline, start here:

- `src/humembr/util/config.py` — configuration dataclass and defaults
- `src/humembr/agent/interview_agent.py` — agent used for question evaluation
- `src/humembr/agent/llm_agent.py` — agent used for real-world robot control
- `src/humembr/agent/tools.py` — tools available to the agents
- `src/humembr/processing/person_processor.py` — feature extraction and clustering
- `src/humembr/processing/qwen.py` — Ollama caption client
- `src/humembr/processing/qwen_openai.py` — vLLM / OpenAI-compatible caption client
