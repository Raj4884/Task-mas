# Qwen2.5 Finetuning Guide

This folder contains a highly optimized script (`finetune_qwen.py`) to finetune Qwen2.5 on your astrology dataset using Unsloth (which provides 2x faster training and 70% less memory usage via QLoRA).

## 1. Preparing the Dataset

1. Download your dataset from the Google Drive link provided.
2. Ensure the dataset is in a format compatible with the script. The script currently expects a JSON file structured like `astrologer_sample_chats.json` (a list of objects, each containing a `"messages"` array of role/content pairs).
3. Place the downloaded dataset in this directory (`d:\Task job`).
4. Open `finetune_qwen.py` and change the `DATASET_PATH` variable to the name of the file you downloaded.

## 2. Environment Setup

If you are running this on a cloud GPU (e.g., Google Colab, RunPod, or a VPS with an Nvidia GPU), you need to install the Unsloth dependencies.

Run this in your terminal:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

## 3. Running the Script

Once the dataset is in place and dependencies are installed, run the script:

```bash
python finetune_qwen.py
```

- The script will automatically download the base model (`Qwen2.5-7B-Instruct-bnb-4bit`).
- It will format your dataset to match Qwen's ChatML template.
- It will train the LoRA adapters.
- Finally, it will save the adapters to a folder named `qwen2.5_astrologer_lora`.

## 4. Next Steps (Hosting)

Once training is complete, you can take the `qwen2.5_astrologer_lora` folder and host it using vLLM! 
Refer to `vllm_hosting_guide.md` for detailed instructions on hosting.
