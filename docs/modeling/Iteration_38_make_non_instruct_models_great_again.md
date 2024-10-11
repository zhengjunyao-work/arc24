# Iteration 38. Make non-instruct models great again

_11-10-2024_

## Goal

Modify the training of non-instruct models so they learn to stop the response at inference.

## Motivation

I have made experiment with non-instruct models at the past, but they do not stop the response at inference
and thus inference times are higher because they repeat the response over and over.

I have evidence that non-instruct models might give better results, but I have to find the way
to train them correctly.

There must be an easy way to fix this.

## Development

### Experiment design

My idea is to fine-tune Qwen2.5-0.5B on a tiny dataset of just 5 samples. I will choose the smaller
samples from the ARC tasks to train faster and have smaller VRAM requirements.

Then I will make inference and see if the model stops the responses or not.

### Updating transformers and accelerate

The python environment is currently a little big unstable due to the installations I did for omni-arc.

I had to update both transformers and accelerate to make the training script work again on my computer.

```bash
pip install --upgrade transformers accelerate
```

### Trainings

<details>
  <summary>Click to see bash commands</summary>

```bash
# baseline
python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 32 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/01_baseline \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose

accelerate launch fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 32 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/01_baseline_accelerate \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose


accelerate launch fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B-Instruct \
--device_map None \
--lora_r 32 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/02_baseline_instruct \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose

accelerate launch fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--no-use_lora \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/03_full-fine-tune \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose


accelerate launch fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--no-use_lora \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/04_full-fine-tune_change_pad_token \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose


accelerate launch fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--no-use_lora \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/05_full-fine-tune_change_pad_token_fix_tokenizer_bug \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose


accelerate launch fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 32 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/06_final_experiment_with_lora \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose

accelerate launch fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 128 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/07_final_experiment_with_lora_longer \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 500 \
--logging_steps 10 \
--batch_size 16 \
--learning_rate 1e-4 \
--verbose

for checkpoint_folder in /mnt/hdd0/Kaggle/arc24/models/20241011_non-instruct_models/*/checkpoint-100; do
    python easy_inference_and_evaluation.py "${checkpoint_folder}" --dataset_path /mnt/hdd0/Kaggle/arc24/data/new_partitions/smaller_5_tasks.json --predictions_per_task 8
done
```

</details>

Training for 100 steps takes 7:46 minutes without accelerate. With accelerate just a little bit more than 2 minutes.

## Results

I have made two improvements to the existing code:

1. Qwen tokenizer does not need to be resized, I just needed to change the `eos_token` to be the same
   as the instruct model.
2. I have added the tokenizer to the train function, that way it is saved in the checkpoint.

The problem was that the original tokenizer was being saved instead of the modified one, thus at inference
there was a discrepancy between the model and the tokenizer.

I have been able to fine-tune the non-instruct version without any problem and make inference correctly.

However it seems that LoRA is not enough for the non-instruct version, if I want to use it I have to
fully fine-tune the model.

## Conclusion

I can now use the non-instruct models, but I have to fully fine-tune them. I cannot just use LoRA.
Thus it is unclear if this will be useful.

## Next steps

## TODO

- [x] Create a small dataset for training and validation
- [x] Train, make inference and verify if it works
