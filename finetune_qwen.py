import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Change this to match the dataset format you downloaded from Google Drive
# If your dataset is a JSON file like the sample, point to it here.
DATASET_PATH = "astrologer_sample_chats.json" 
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" # Pre-quantized for memory efficiency
MAX_SEQ_LENGTH = 2048 # Adjust based on GPU memory
OUTPUT_DIR = "qwen2.5_astrologer_lora"

def main():
    print(f"Loading Model: {MODEL_NAME}")
    
    # 1. Load the model and tokenizer using Unsloth (highly optimized)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, # Auto-detect (bfloat16 for modern GPUs)
        load_in_4bit=True, # QLoRA - reduces memory usage drastically
    )

    # 2. Add LoRA Adapters (This makes finetuning fast and efficient)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing="unsloth", # Crucial for saving VRAM
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Setup the Chat Template
    # Qwen uses ChatML template natively. We format our raw data using this template.
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml", 
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
    )

    def formatting_prompts_func(examples):
        conversations = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in conversations]
        return { "text" : texts }

    # 4. Load the Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please ensure you downloaded it from Google Drive.")

    # Assuming the drive dataset is formatted similarly to the sample JSON (list of {"messages": [...]})
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    print("Formatting dataset...")
    dataset = dataset.map(formatting_prompts_func, batched=True,)

    # 5. Setup the Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False, # Can be True for speed if sequences are short
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3, # Adjust based on dataset size (more epochs for small datasets)
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none", # Change to "wandb" if you use weights and biases
        ),
    )

    # 6. Start Training
    print("Starting Training...")
    trainer_stats = trainer.train()
    print(f"Training Complete. Stats: {trainer_stats}")

    # 7. Save the Model
    print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! You can now serve this using vLLM or merge it.")

if __name__ == "__main__":
    main()
