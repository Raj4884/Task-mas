# Guide: Hosting a Finetuned Model on a VPS using vLLM

vLLM is a highly optimized library for LLM inference. It provides excellent throughput and an OpenAI-compatible API server, making it a perfect choice for hosting your finetuned Qwen model.

## Prerequisites

1. **VPS Specifications:**
   - **GPU:** You need an Nvidia GPU with sufficient VRAM (e.g., RTX 3090/4090 with 24GB VRAM, or an A10/A100 depending on model size). A 7B model quantized or with 16-bit precision requires roughly 16GB-24GB of VRAM.
   - **OS:** Ubuntu 22.04 or 20.04 LTS is highly recommended.
   - **RAM:** At least 32GB system RAM.
   - **Storage:** 50GB+ NVMe SSD (for weights and caching).

2. **Required Software:**
   - Python 3.10+
   - Nvidia Drivers
   - CUDA Toolkit (CUDA 12.1+ is standard for modern vLLM)

---

## Step 1: Setting up the VPS Environment

Once you SSH into your VPS, ensure your packages are up to date and install the basic utilities:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

### Install Nvidia Drivers & CUDA Toolkit (if not pre-installed)
Most GPU VPS providers (like RunPod, Lambda Labs, or AWS) come with an image that already has CUDA installed. You can verify this by running:
```bash
nvidia-smi
```
If you see a table with your GPU name and CUDA version, you are good to go.

---

## Step 2: Install vLLM

It's best to create a virtual environment to avoid conflicts.

```bash
# Create a virtual environment
python3 -m venv vllm-env
source vllm-env/bin/activate

# Install vLLM (this will also pull torch and other dependencies)
pip install vllm
```

---

## Step 3: Preparing the Finetuned Model

Before hosting, you need your model weights. If you finetuned your model using LoRA, you must **merge the LoRA weights** with the base model and save the full model, OR you can serve the base model and pass the LoRA adapters dynamically (vLLM supports this).

### Option A: Hosting a Merged Model (Recommended for simplicity)
Upload your merged model directory (containing `config.json`, `model.safetensors`, `tokenizer.json`, etc.) to your VPS. Let's say it's located at `/home/user/my-qwen-finetuned/`.

### Option B: Hosting via HuggingFace Hub
If you uploaded your finetuned model to HuggingFace, you can use the repository ID (e.g., `yourusername/Qwen2.5-Astrology-Finetune`).

---

## Step 4: Starting the vLLM API Server

vLLM provides an OpenAI-compatible API out of the box. Start the server using the following command:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /home/user/my-qwen-finetuned/ \
    --served-model-name qwen2.5-astrologer \
    --host 0.0.0.0 \
    --port 8000
```

> [!TIP]
> **Important Flags:**
> - `--host 0.0.0.0`: Exposes the server to the internet (make sure your VPS firewall allows port 8000).
> - `--max-model-len 4096`: Qwen models support long contexts, but limiting this can save VRAM.
> - `--dtype bfloat16`: Highly recommended to reduce memory usage without losing precision.
> - `--enable-lora`: Use this flag if you are serving a base model but want to dynamically load LoRA weights.

---

## Step 5: Testing the Server

Once the server says "Uvicorn running on http://0.0.0.0:8000", you can test it from another terminal or your local computer.

```bash
curl http://YOUR_VPS_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-astrologer",
    "messages": [
      {"role": "system", "content": "You are a knowledgeable and empathetic astrologer."},
      {"role": "user", "content": "I am feeling very lost in my career right now. Can you help me?"}
    ]
  }'
```

---

## Step 6: Running the Server in the Background

If you close your SSH session, the server will stop. To keep it running, use `tmux` or `systemd`.

### Using tmux:
```bash
tmux new -s vllm_server
# (Run the vllm python command here)
# Press Ctrl+B, then D to detach. 
# To reattach later: tmux attach -t vllm_server
```

### Using systemd (For Production):
Create a service file:
```bash
sudo nano /etc/systemd/system/vllm.service
```
Add the following (adjust paths!):
```ini
[Unit]
Description=vLLM API Server
After=network.target

[Service]
User=your_username
WorkingDirectory=/home/your_username
ExecStart=/home/your_username/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server --model /path/to/model --served-model-name qwen2.5-astrologer --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
Start and enable it:
```bash
sudo systemctl daemon-reload
sudo systemctl start vllm
sudo systemctl enable vllm
```

## Step 7: Security (Critical for Production)

If exposing `0.0.0.0`, anyone can access your API. You should:
1. Put the API behind an Nginx reverse proxy.
2. Add a Bearer Token for authentication using vLLM's built-in argument: `--api-key YOUR_SECRET_KEY`.
3. Use HTTPS via Certbot/Let's Encrypt.
