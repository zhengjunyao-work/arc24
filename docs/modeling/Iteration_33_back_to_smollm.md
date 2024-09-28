# Iteration 33. Back to SmolLM

_start date_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Train with SmolLM models to see if we can reach similar accuracy to Qwen but with faster models.

## Motivation

I have recently tried the new [Llama 3.2 1B](Iteration_32_llama_32.md) and it was better than Qwen but slower.
I have the intuition that a small model trained for longer could reach the same accuracy as a bigger model.
But this smaller model could be test-time fine-tuned for more steps or do more predictions.

## Development

In the [SmolLM blog](https://huggingface.co/blog/smollm) they say the following:

> For all three models we use embedding tying and a context length of 2048 tokens. This context length can be further extended with some long context fine-tuning.

Let's see if we can really train the models with a bigger context length and they work well at inference.

I'm going to go directly for the smaller model `SmolLM-135M-Instruct` because there is a 360M parameter
model but that is very close to Qwen's 500M.

### Tokenizer analysis

### Local experiments

<details>
  <summary>Click to see bash commands</summary>

```bash
# baseline, 492 seconds, 4.9 seconds/it
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM/01_baseline \
--max_seq_len 10240 \
--device_map None \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4

# Try to increase per_device_train_batch_size but get OOM
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM/02_bs2 \
--max_seq_len 10240 \
--device_map None \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4 \
--per_device_train_batch_size 2

# train on a single gpu, 338s, this uses ~21GB of VRAM, 3.3 seconds per iteration
export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM/03_1gpu \
--max_seq_len 10240 \
--device_map None \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4

# Reduce the msl to 2048, now it only uses 7GB of VRAM, 294s, 2.9 seconds per iteration
export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM/04_1gpu_2048msl \
--max_seq_len 2048 \
--device_map None \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4

# 186 seconds, 1.8 seconds per step
export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM/05_1gpu_2048msl_pdbs2 \
--max_seq_len 2048 \
--device_map None \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4 \
--per_device_train_batch_size 2
```

</details>

It is training at a speed of 1.8 seconds per step on a single GPU and with a max_seq_len of 2048.
For reference Qwen trained at 6 seconds per step and Llama at 9 when being trained on 2 gpus.
So potentially we are looking at a speedup of 6-7. If we are able to train SmolLM to a similar accuracy
to Qwen this would be game changing.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
