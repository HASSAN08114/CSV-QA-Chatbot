import torch
import json
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Dict, Any

class QLoRATrainer:
    """
    QLoRA trainer for fine-tuning models on CSV Q&A data
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium", device="auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        
    def setup_quantization(self):
        """Setup 4-bit quantization for memory efficiency"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        return bnb_config
    
    def setup_lora_config(self):
        """Setup LoRA configuration for parameter-efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        return lora_config
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization and LoRA"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model with quantization...")
        bnb_config = self.setup_quantization()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        
        print("Applying LoRA...")
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, training_data: List[Dict], max_length: int = 512):
        """Prepare dataset for training"""
        print(f"Preparing dataset from {len(training_data)} samples...")
        
        # Split data
        train_data, val_data = train_test_split(
            training_data, test_size=0.2, random_state=42
        )
        
        # Tokenize function
        def tokenize_function(examples):
            # Combine input and target
            combined_texts = [
                f"{input_text}\n\nAnswer: {target_text}{self.tokenizer.eos_token}"
                for input_text, target_text in zip(examples["input_text"], examples["target_text"])
            ]
            
            tokenized = self.tokenizer(
                combined_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
        
        return train_dataset, val_dataset
    
    def setup_training_args(self, output_dir="./fine_tuned_model", num_epochs=3):
        """Setup training arguments"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=25,
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if wandb.run is not None else None,
            run_name="csv-qa-qlora-finetuning",
            save_total_limit=2,
            dataloader_pin_memory=False,
        )
        return training_args
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Remove padding tokens
        true_predictions = [
            [p for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for l in label if l != -100]
            for label in labels
        ]
        
        # Calculate accuracy
        correct = 0
        total = 0
        for pred, label in zip(true_predictions, true_labels):
            if len(pred) == len(label):
                correct += sum(1 for p, l in zip(pred, label) if p == l)
                total += len(label)
        
        accuracy = correct / total if total > 0 else 0
        
        return {"accuracy": accuracy}
    
    def train(self, training_data: List[Dict], output_dir="./fine_tuned_model"):
        """Main training function"""
        print("Starting QLoRA fine-tuning...")
        
        # Initialize wandb
        try:
            wandb.init(project="csv-qa-finetuning", name="qlora-experiment")
        except:
            print("Wandb not available, continuing without logging")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Prepare dataset
        train_dataset, val_dataset = self.prepare_dataset(training_data)
        
        # Setup training arguments
        training_args = self.setup_training_args(output_dir)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving model...")
        trainer.save_model()
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Log final metrics
        final_metrics = trainer.evaluate()
        print(f"Final metrics: {final_metrics}")
        
        if wandb.run is not None:
            wandb.log(final_metrics)
            wandb.finish()
        
        return trainer, model

def quick_train_from_collected_data(
    data_file="training_data/training_data.jsonl",
    model_name="microsoft/DialoGPT-medium",
    output_dir="./fine_tuned_model",
    min_samples=20
):
    """
    Quick training function that loads collected data and starts training
    """
    # Load training data
    if not os.path.exists(data_file):
        print(f"Training data file {data_file} not found!")
        print("Please collect some data first by using the chat interface.")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        training_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(training_data)} training samples")
    
    if len(training_data) < min_samples:
        print(f"Need at least {min_samples} samples, but only have {len(training_data)}")
        print("Please collect more data by using the chat interface.")
        return None
    
    # Start training
    trainer = QLoRATrainer(model_name=model_name)
    trainer_instance, model = trainer.train(training_data, output_dir)
    
    print(f"Training completed! Model saved to {output_dir}")
    return trainer_instance, model

if __name__ == "__main__":
    # Quick training from command line
    quick_train_from_collected_data()
