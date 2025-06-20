
# SMDM-dev

## Install

```bash
pip install uv
uv venv
uv sync --no-build-isolation
source .venv/bin/activate
```

## Preprocess data

Make sure your data in the `--source_path` is in the following format:
```
data/slimpajama/
├── train
│   ├── chunk1
│   ├── chunk2
│   ├── ...
├── validation
│   ├── chunk1
│   ├── chunk2
│   ├── ...
└── test
    ├── chunk1
    ├── chunk2
```

```bash
bash run_process_slimpajama.sh
```
