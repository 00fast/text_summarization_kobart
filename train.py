import torch
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from dataset import MyDataset, process_file
from transformers import AutoTokenizer
import glob

def train_model():
    # file_path
    file_paths = glob.glob('/yourpath/TL_인공물_ED/*.json') 

    # Process all JSON files and filter out the None values
    context_summary_pairs = [pair for file_path in file_paths for pair in process_file(file_path)]
    contexts, summaries = zip(*context_summary_pairs)
    
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("digit82/kobart-summarization")

    # Load the previously trained weights (if needed)
    # model.load_state_dict(torch.load('weights.ckpt'))  # Path to the previously trained weights

    # Cut sequences longer than the maximum length of the model and padded to the maximum length
    inputs = tokenizer(contexts, truncation=True, padding=True, max_length=512)
    labels = tokenizer(summaries, truncation=True, padding=True, max_length=128)

    # dataset 
    dataset = MyDataset(inputs, labels)

    # traing arguments
    training_args = TrainingArguments(
        output_dir='./new_results',          
        num_train_epochs=10,                 
        per_device_train_batch_size=16,      # batch size
        per_device_eval_batch_size=64,       # batch size
        warmup_steps=500,                    # Used to incrementally increase the learning rate
        weight_decay=0.01,                   # strength of weight decay
        logging_dir='./new_logs',            
        logging_steps=10,                    # Learning Loss Tracking
        report_to=None,                      # Without WandB
    )

    # Create the Trainer and train
    trainer = Trainer(
        model=model,                         
        args=training_args,                 
        train_dataset=dataset,              
    )

    trainer.train()

    # Save the newly trained model weights
    torch.save(model.state_dict(), 'weights.ckpt')