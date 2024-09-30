# Iteration 33. Back to SmolLM

_28-09-2024_

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

### How to increase the context length

- https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/discussions/8
- [ChatGPT suggestions to increase the context length](https://chatgpt.com/share/66f7d739-2f2c-8012-bcff-be2ec4c14da7)
- [More educated information from ChatGPT](https://chatgpt.com/share/66f7df6a-555c-8012-ac1b-b643a3627937)
- https://gradient.ai/blog/scaling-rotational-embeddings-for-long-context-language-models
- https://blog.eleuther.ai/rotary-embeddings/
- [Is it true that context window can be safely doubled without repercussions? #7206](https://github.com/ggerganov/llama.cpp/discussions/7206)
- https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig.rope_scaling
- Configuration samples:
  - [Qwen2.5B, 32k context, theta 1000000](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json#L19)
  - [Phi-3-mini-4k-instruct, 4k context, theta 10000.0](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json)
  - [Phi-3-mini-128k-instruct, 131k context, theta 10000.0, uses longrope](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json)
  - [SmolLM-135M-Instruct, 2k context, theta 10000.0](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct/blob/main/config.json)
  - [Llama-3.1-8B-Instruct, 131k context, theta 500000, original context 8k but uses llama3 rope_scaling](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json)
- https://wandb.ai/byyoung3/ml-news/reports/Scaling-Llama-2-to-32k-Tokens-With-LongLora--Vmlldzo1NzU4OTk2

It seems that theta determines the original context. If a longer context is needed it seems that all the people
use the rope_scaling.

<details>
  <summary>Click to see bash commands</summary>

```bash
# train on a single gpu, 338s, this uses ~21GB of VRAM, 3.3 seconds per iteration
export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--no-use_lora \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM_context_window/01_baseline-full-fine-tuning \
--max_seq_len 10240 \
--device_map None \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 4e-4

export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--no-use_lora \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM_context_window/02_change-model-config \
--max_seq_len 10240 \
--device_map None \
--max_steps 100 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 4e-4

export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--no-use_lora \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM_context_window/03_change-model-config-longer \
--max_seq_len 10240 \
--device_map None \
--max_steps 1000 \
--warmup_ratio 1e-1 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--random_seed 7 \
--learning_rate 4e-4

export CUDA_VISIBLE_DEVICES=1
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--no-use_lora \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM_context_window/04_longer-baseline \
--max_seq_len 10240 \
--device_map None \
--max_steps 1000 \
--warmup_ratio 1e-1 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--random_seed 7 \
--learning_rate 4e-4

export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--no-use_lora \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM_context_window/05_rope-scaling-02 \
--max_seq_len 10240 \
--device_map None \
--max_steps 1000 \
--warmup_ratio 1e-1 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--random_seed 7 \
--learning_rate 4e-4

python fine-tuning.py \
--model_path /home/gbarbadillo/data/SmolLM-135M-Instruct \
--n_gpus 1 \
--no-use_lora \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240928_debug_SmolLM_context_window/07_linear-rope-scaling-2-update-tokenizer \
--max_seq_len 10240 \
--device_map None \
--max_steps 1000 \
--warmup_ratio 1e-1 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--random_seed 7 \
--learning_rate 4e-4
```

</details>

### AMD-Llama-135m

This model encodes each number independently, just like SmolLM.

It is not an instruct model and it does not have a chat template.

https://huggingface.co/docs/transformers/main/en/chat_templating#advanced-adding-and-editing-chat-templates

### Current problems

`model_max_length` in tokenizer, does not seem to be saved correctly, I have manually fixed it.

It makes very long predictions, just like non-instruct Qwen models.

The baseline model that was simply fine-tuned does not work with VLLM because the following error:

> ValueError: User-specified max_model_len (10240) is greater than the derived max_model_len (max_position_embeddings=2048 or model_max_length=None in model's config.json). This may lead to incorrect model outputs or CUDA errors. To allow overriding this maximum, set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

If I modify the model configuration to increase the max_model_len it works, but it seems to be predicting
special tokens all the time because the predictions appear to be empty, but it takes 3599s to do the predictions.
If I modify the inference script to just use 2048 it does the same thing with the predictions but faster.
Thus it appears that without modifications the model cannot work correctly.

## Results

### Training metrics

As a reference when training Qwen or Llama for 10k steps I could reach a train and validation loss around 0.08.
Training for 80k steps could reduce the train loss to 0.03, but the validation loss did not improve.

## Conclusion

## Next steps

- After fixing the problem with the tokenizer, I could now train Qwen2.5 non-instruct model and have fast inference.

## TODO

- [x] What is the speedup when training?
- [x] Train a model for 10k steps to find what is the optimal learning rate -> 8e-4
- [ ] Does the evaluation return comparable metrics to Qwen?
- [ ] What is the speedup at inference?
- [ ] Try to get the same metrics as Qwen by training for much longer, f.e. 160k steps
- [ ] Another small LLM, it also has 2048 context window: https://huggingface.co/amd/AMD-Llama-135m