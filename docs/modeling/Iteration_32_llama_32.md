# Iteration 32. Llama 3.2

_26-09-2024_

## Goal

Try the new Llama 3.2 1B model

## Motivation

Meta has released a new Llama 3.2 1B model. The size is twice the size of Qwen-0.5B but I believe
it is worth trying.

## Development

I have to explore what is the best way to encode the grids for this Llama model.

- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Tokenizer analysis

Exploring how the tokenizer works is crucial to select a good grid encoding. We want to use an
encoding that uses different symbols for each cell in the grid and at the same time that it is able
to represent the grid with the minimum number of tokens possible.

Llama 3.2 1B uses the same tokenizer as Llama 3.1 7B. This implies that it has words for each number
from 0 to 999. In contrast Qwen uses a different word for each number, making it easier to represent
the grids. One solution is to repeat each number 3 times to encode the grid for Llama.

### Local trainings

<details>
  <summary>Click to see bash commands</summary>

```bash
python fine-tuning.py \
--model_path /home/gbarbadillo/data/Llama-3.2-1B-Instruct \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(RepeatNumberEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240926_debug_Llama32/01_baseline \
--max_seq_len 2048 \
--device_map None \
--max_steps 500 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4

python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B-Instruct \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240926_debug_Llama32/02_qwen_25 \
--max_seq_len 2048 \
--device_map None \
--max_steps 500 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4

python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2-0.5B-Instruct \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240926_debug_Llama32/03_qwen_2 \
--max_seq_len 2048 \
--device_map None \
--max_steps 500 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4
```

</details>

I had to decrease the max_seq_len to 2048 to be able to train on my computer. Each step was taking around 9 seconds.
For Qwen2.5 it takes around 6 seconds per step. The token length distribution is the same despite using different grid encoders.
Qwen2 has the same speed as Qwen2.5

**So the Llama-1B model is 50% slower at training than Qwen-0.5B and is more memory hungry.**

## Results

| model                 | accuracy   | pass_32    | vote_2     | vote_1     |
|-----------------------|------------|------------|------------|------------|
| Qwen2-0.5B            | 8.24%      | 26.50%     | 15.91%     | 13.01%     |
| Qwen2-0.5B-Instruct   | 8.25%      | 26.75%     | 15.91%     | 13.01%     |
| Qwen2.5-0.5B          | 9.37%      | 26.75%     | 18.31%     | **15.28%** |
| Qwen2.5-0.5B-Instruct | 8.98%      | 26.00%     | 17.93%     | 14.27%     |
| Llama-3.2-1B          | **10.25%** | **29.00%** | **19.88%** | 14.88%     |

Validation results show an improvement when fine-tuning Llama for ARC tasks over Qwen2.5 and Qwen2.

Inference took 2473 seconds, compared to the 1809 seconds that takes inference with Qwen. Inference is 37% slower.

## Conclusion

Llama-3.2-1B gets better validation results than Qwen but at the cost of being 50% slower on training and 37% slower on inference.
It is unclear if it is worth it. I would rather try using a model smaller than Qwen-0.5B to have more
room for test-time fine-tuning and inference.

## Next steps

## TODO

- [ ] Read about the new release. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- [x] What is the best way to encode the grids? Check the tokenizer. Maybe create a notebook to do this inspection, it will be useful in the future if more small models are released.
- [x] What is the optimal learning rate?
- [x] How is the training and inference speed compared to Qwen2.5-0.5B? Do I have to reduce the number of predictions or the duration of the test-time fine-tuning?