# Explore Various sparse attentions in chat mode.

### Install sparse attention hub

```
git clone https://github.com/skylight-org/sparse-attention-hub.git
cd sparse-attention-hub
pip install -e . && pip install -e .[dev]

### starting the API
python scripts/start_server.py [model name] [port] --config dense

Example: python scripts/start_server.py Qwen/Qwen3-Coder-30B-A3B-Instruct 4000 --config dense

### running chat
python demo/apiChat.py


