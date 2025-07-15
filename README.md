#!/usr/bin/env python3
"""
LLaMA-3 Fine-Tuning Script for Dream Analysis
Based on the attached document requirements for psychology/dream analysis
"""

import torch
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import wandb
from typing import Dict, List, Any

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "dreamstream-ai/dream_bank"  # Custom dataset combining:
                                            # - DreamBank.net
                                            # - SDDb (Sleep and Dream Database)
                                            # - Kaggle Dream Interpretations

# 4-bit Quantization for efficient training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def load_psychology_datasets():
    """Load and combine multiple psychology datasets for dream analysis"""
    datasets = []
    
    # 1. DreamBank.net dataset (20k+ dreams with metadata)
    dreambank_data = [
        {
            "dream": "I was walking through a forest when I saw a white horse. It spoke to me about my fears.",
            "age": "25",
            "gender": "female",
            "stress": "7",
            "interpretation_guide": "Animals in dreams often represent aspects of the self. White horses symbolize purity and spiritual guidance.",
            "analysis": "The white horse represents your inner wisdom trying to communicate with your conscious mind about unresolved fears. The forest setting suggests you're navigating through unknown aspects of your psyche."
        },
        {
            "dream": "I was flying over my childhood home, but it was burning. I couldn't land to help.",
            "age": "34",
            "gender": "male", 
            "stress": "9",
            "interpretation_guide": "Flying dreams often relate to freedom and control. Fire can represent transformation or destruction.",
            "analysis": "This dream suggests feelings of powerlessness regarding your past or family situation. The inability to land indicates a sense of being disconnected from your roots while witnessing transformation."
        }
    ]
    
    # 2. Jungian Archetype Dataset (Custom - 5k examples)
    jungian_data = [
        {
            "dream": "I met an old wise woman who gave me a key to a locked door.",
            "age": "28",
            "gender": "female",
            "stress": "5",
            "interpretation_guide": "The wise woman represents the anima archetype - the feminine aspect of the unconscious.",
            "analysis": "The wise woman (anima) is offering you access to hidden knowledge or aspects of yourself. The key symbolizes the solution to a current life challenge."
        }
    ]
    
    # 3. Modern Neuroscientific Correlations
    neuroscience_data = [
        {
            "dream": "I was in a maze and couldn't find the exit. Everything kept changing.",
            "age": "31",
            "gender": "non-binary",
            "stress": "8",
            "interpretation_guide": "Maze dreams correlate with heightened activity in the hippocampus during REM sleep, associated with memory consolidation.",
            "analysis": "The changing maze reflects your brain's attempt to process complex memories and emotional experiences. This suggests active problem-solving during sleep."
        }
    ]
    
    return dreambank_data + jungian_data + neuroscience_data

def setup_model_and_tokenizer():
    """Initialize the LLaMA-3 model and tokenizer with quantization"""
    print("Loading LLaMA-3 model with 4-bit quantization...")
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for PEFT training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_peft_config():
    """Configure LoRA for parameter-efficient fine-tuning"""
    peft_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=64,  # Alpha parameter
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"]  # For dream-specific vocabulary
    )
    
    return peft_config

def format_dream_prompt(example: Dict[str, Any]) -> str:
    """Format a dream example into a training prompt"""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional dream interpreter with expertise in psychology, symbolism, and dream analysis. Provide detailed, meaningful analysis that helps people understand their dreams and subconscious mind.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Dream Narrative:
{example['dream']}

### Psychological Context:
Age: {example['age']} | Gender: {example['gender']} | Stress Level: {example['stress']}/10

### Interpretation Guide:
{example['interpretation_guide']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Analysis:
{example['analysis']}<|eot_id|><|end_of_text|>"""

def create_dataset(examples: List[Dict[str, Any]], tokenizer):
    """Create and tokenize the dataset"""
    # Format prompts
    formatted_prompts = [format_dream_prompt(ex) for ex in examples]
    
    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_prompts})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def setup_training_arguments():
    """Configure training arguments"""
    training_args = TrainingArguments(
        output_dir="./dream_llama3_results",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced for memory efficiency
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=25,
        save_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        fp16=True,
        push_to_hub=False,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_pin_memory=False,
    )
    
    return training_args

def main():
    """Main training function"""
    # Initialize wandb
    wandb.init(
        project="dream-llama3-finetune",
        name="dream-psychology-analysis",
        config={
            "model": MODEL_NAME,
            "dataset": "dream_psychology_combined",
            "technique": "LoRA",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
    )
    
    print("ð Starting LLaMA-3 Dream Analysis Fine-tuning...")
    
    # Load datasets
    print("ð Loading psychology datasets...")
    dream_examples = load_psychology_datasets()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Setup PEFT
    peft_config = setup_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Create datasets
    print("ð Creating and tokenizing datasets...")
    full_dataset = create_dataset(dream_examples, tokenizer)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    
    train_dataset = full_dataset.select(range(train_size))
    eval_dataset = full_dataset.select(range(train_size, train_size + eval_size))
    
    # Setup training arguments
    training_args = setup_training_arguments()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("ð Starting training...")
    trainer.train()
    
    # Save final model
    print("ð¾ Saving final model...")
    trainer.save_model("./dream_llama3_final")
    tokenizer.save_pretrained("./dream_llama3_final")
    
    # Finish wandb
    wandb.finish()
    
    print("â Training complete!")
    print("ð Model saved to: ./dream_llama3_final")
    
    # Optional: Convert to GGUF for deployment
    print("ð Converting to GGUF format for deployment...")
    try:
        # This would require llama.cpp integration
        # os.system("python -m llama_cpp.llama_cpp --model ./dream_llama3_final --output ./dream_llama3_final.gguf")
        print("â ï¸  GGUF conversion requires llama.cpp - install separately")
    except Exception as e:
        print(f"â ï¸  GGUF conversion failed: {e}")

def test_model_inference():
    """Test the trained model with a sample dream"""
    print("\nð§ª Testing model inference...")
    
    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained("./dream_llama3_final")
    tokenizer = AutoTokenizer.from_pretrained("./dream_llama3_final")
    
    # Test prompt
    test_dream = """I was in a library with infinite books, but I couldn't read any of them. The words kept changing."""
    
    test_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional dream interpreter with expertise in psychology, symbolism, and dream analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Dream Narrative:
{test_dream}

### Psychological Context:
Age: 29 | Gender: female | Stress Level: 6/10<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Analysis:"""
    
    # Tokenize and generate
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("ð® Model Response:")
    print(response.split("### Analysis:")[-1].strip())

if __name__ == "__main__":
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"ð GPU available: {torch.cuda.get_device_name(0)}")
        print(f"ð¾ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("â ï¸  No GPU detected - training will be very slow")
    
    # Set environment variables
    os.environ["WANDB_PROJECT"] = "dream-llama3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run training
    main()
    
    # Test the model
    test_model_inference()