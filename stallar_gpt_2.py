from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
#using gpt-2 to train my model
import os
csv_path = "space_dataset_merged.csv"   #dataset was created by myself(sanchit)
text_column = "response"                
model_name = "gpt2"
output_dir = "./gpt2-finetuned"

#Loading created data 
dataset = load_dataset("csv", data_files={"train": csv_path})

#breaking the data into token
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
model = GPT2LMHeadModel.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        truncation=True,
        padding="max_length",
        max_length=128
    )
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  
    save_steps=200,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=True,               
    report_to="none",                 
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()

#saving the model to be used in future
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved in {output_dir}")
generator = pipeline("text-generation", model=output_dir, tokenizer=output_dir)
#test the training with a prompt
print(generator("Your custom prompt: ", max_length=100, num_return_sequences=1))
