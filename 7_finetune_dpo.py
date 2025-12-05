#!/usr/bin/env python3
"""
Fine-tune LLM with DPO (Direct Preference Optimization) + LoRA

This script fine-tunes a language model using DPO to prefer "chosen" 
responses over "rejected" ones, creating a model that better matches
philosopher response patterns.

Supports multiple base models:
- Qwen/Qwen2.5-0.5B-Instruct (for testing)
- meta-llama/Llama-3.1-8B-Instruct (for production)

Requires:
- philosopher_dpo_train.jsonl (from 8_prepare_dpo_dataset.py)
- philosopher_dpo_val.jsonl

Outputs:
- ./llama_philosopher_dpo/final_model (LoRA adapter weights)
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings('ignore')
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import os

# ============================================================================
# CONFIGURATION - Modify these for your use case
# ============================================================================

# Model options:
# - "Qwen/Qwen2.5-0.5B-Instruct" (small, for testing)
# - "meta-llama/Llama-3.1-8B-Instruct" (production)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Output directory for trained model
OUTPUT_DIR = "./qwen2.5_0.5b_philosopher_dpo"

# Training data files
TRAIN_FILE = "philosopher_dpo_train.jsonl"
VAL_FILE = "philosopher_dpo_val.jsonl"

# Training settings
USE_FULL_DATASET = False  # Set to True for production training
NUM_EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
BETA = 0.1  # DPO temperature parameter

# For testing, limit number of examples
TEST_TRAIN_EXAMPLES = 50
TEST_VAL_EXAMPLES = 10

# HuggingFace token (set if using gated models like Llama)
HF_TOKEN = None  # or set to your token


def load_jsonl(filepath):
    """Load JSONL dataset"""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def prepare_dpo_dataset(examples):
    """
    Prepare dataset for DPO training
    DPO expects: prompt, chosen, rejected (all strings)
    """
    return examples


def main():
    print("=" * 80)
    print("DPO Fine-Tuning with LoRA")
    print("=" * 80)

    # Check device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("\n✓ Using Metal (MPS) backend")
    elif torch.cuda.is_available():
        device = "cuda"
        print("\n✓ Using CUDA backend")
    else:
        device = "cpu"
        print("\n⚠️  Using CPU backend (training will be slow)")

    print(f"\n1. Loading model: {MODEL_NAME}")

    # Load model
    model_kwargs = {
        "torch_dtype": torch.float16 if device in ["mps", "cuda"] else torch.float32,
        "device_map": None,
        "low_cpu_mem_usage": True
    }
    
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    tokenizer_kwargs = {}
    if HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"   Model loaded: {model.num_parameters():,} parameters")

    # Configure LoRA
    print("\n2. Configuring LoRA for DPO...")
    
    # Adjust target modules based on model architecture
    if "qwen" in MODEL_NAME.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA BEFORE moving to device
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"   LoRA enabled: {trainable_params:,} trainable params ({100 * trainable_params / all_params:.2f}%)")

    # Move model to device after LoRA is applied
    if device == "cuda":
        model = model.cuda()
        print(f"   Model moved to: {device}")
    elif device == "mps":
        model = model.to(device)
        print(f"   Model moved to: {device}")

    # Load DPO datasets
    print("\n3. Loading DPO datasets...")
    
    if not os.path.exists(TRAIN_FILE):
        print(f"\n❌ Error: {TRAIN_FILE} not found!")
        print("   Please run 8_prepare_dpo_dataset.py first")
        return
        
    train_data = load_jsonl(TRAIN_FILE)
    
    if os.path.exists(VAL_FILE):
        val_data = load_jsonl(VAL_FILE)
    else:
        print(f"   Warning: {VAL_FILE} not found, using subset of train for validation")
        val_data = train_data[:20]
        train_data = train_data[20:]

    # Select subset for testing or use full dataset
    if USE_FULL_DATASET:
        train_subset = train_data
        val_subset = val_data
        print(f"   Using FULL dataset")
    else:
        train_subset = train_data[:TEST_TRAIN_EXAMPLES]
        val_subset = val_data[:TEST_VAL_EXAMPLES]
        print(f"   Using SMALL subset for testing")

    print(f"   Training examples: {len(train_subset)}")
    print(f"   Validation examples: {len(val_subset)}")

    # Prepare datasets
    print("\n4. Preparing DPO format...")
    train_formatted = prepare_dpo_dataset(train_subset)
    val_formatted = prepare_dpo_dataset(val_subset)

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    print("   Datasets in correct string format for DPO")

    # Training configuration
    print("\n5. Setting up DPO training configuration...")

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=device == "cpu",
        fp16=device == "cuda",
        bf16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        dataloader_pin_memory=False if device == "mps" else True,
        beta=BETA,
        max_length=2048,
        max_prompt_length=1024,
    )

    print(f"   Training for {training_args.num_train_epochs} epochs")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Beta (DPO temp): {training_args.beta}")

    # Initialize DPO Trainer
    print("\n6. Initializing DPO Trainer...")

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("   DPO Trainer initialized")

    # Train
    print("\n7. Starting DPO training...")
    print("   DPO learns from preference pairs (chosen vs rejected)")
    print("-" * 80)

    try:
        dpo_trainer.train()
    except Exception as e:
        print(f"\n⚠️  Training error: {e}")
        print("\nTroubleshooting:")
        print("  1. Try reducing batch size")
        print("  2. Check if DPO dataset is properly formatted")
        print("  3. Ensure TRL library is installed: pip install trl")
        raise

    print("-" * 80)
    print("\n8. Training complete!")

    # Save model
    final_model_path = f"{OUTPUT_DIR}/final_model"
    print(f"\n9. Saving DPO model to {final_model_path}")
    dpo_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\n" + "=" * 80)
    print("✅ DPO Fine-tuning complete!")
    print("=" * 80)
    print(f"\nModel saved to: {final_model_path}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate with 3_model_eval.py (set USE_FINETUNED_MODEL=True)")
    print(f"  2. Compare with 8_compare_finetuning.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

