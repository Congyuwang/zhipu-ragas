# Run Rags with Zhipu AI

## Steps

1. Clone the repo and init submodules

```bash
git clone https://github.com/Congyuwang/zhipu-ragas.git
cd zhipu-ragas
git submodule update --init --recursive
```

2. Prepare Zhipu API key.

Create a file `.env` in repo root and write in it your api

```bash
ZHIPUAI_API_KEY=your_api_key
```

3. Prepare environment and install langchain

- create venv

```bash
python -m venv venv
```

- install langchain

```bash
cd langchain/libs
cd core
pip install -e .
cd ../community
pip install -e .
cd ../../..
```

- install other dependencies

```bash
pip install cachetools python-dotenv datasets ragas
```

4. Run the code

```bash
cd src
python eval_zhi.py
```
