# Iteration 24. Overfit to the train set

_14-09-2024_

## Goal

If I want to solve the private test set I first have to solve the train set.

## Motivation

I have evaluated the the evaluation dataset, which I use to train the models for submission and the
accuracy was just 18%, and vote_2 solved only 29.5% of the problems. MindsAI team is solving 46% of the
hidden test problems, and I cannot solve more than 30% of the training problems.

If I can train a model to learn well the training data, I could use it to generate more training samples like
the RE-ARC dataset.

## Development

### Implement the option to do full-model fine-tuning

```bash
python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--no-use_lora \
--learning_rate 2e-5 \
--warmup_ratio 0.1 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_full_fine-tuning/01_baseline-no-lora_lr2e-5 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 32 \
--learning_rate 1e-4 \
--warmup_ratio 0.1 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_full_fine-tuning/02_lora-r-32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5
```

### Multi-gpu training

#### Local experiments

I'm training models for 100k steps and it seems it's going to take around 100 hours. If I can speedup
this time using multiple gpus it would be very helpful to enable faster experimentation.

I'm going to do local experiments with one and two gpus to see if I can benefit from faster training.

```bash
export CUDA_VISIBLE_DEVICES=0
python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 32 \
--learning_rate 1e-4 \
--warmup_ratio 0.1 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240918_debug_multi-gpu/01_baseline-1-gpu \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=50 \
--logging_steps=5 \
--random_seed=7 \
--n_gpus=1 \
--device_map None \
--no-verbose \
--batch_size=16

unset CUDA_VISIBLE_DEVICES
accelerate launch fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 32 \
--learning_rate 1e-4 \
--warmup_ratio 0.1 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240918_debug_multi-gpu/02-2-gpus-accelerate \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=50 \
--logging_steps=5 \
--random_seed=7 \
--n_gpus=2 \
--device_map None \
--no-verbose \
--batch_size=16
```

The training loss is exactly the same, but the training time is 89s when using 2 gpus vs 154 seconds
when using a single gpu.

There were some problems with doble logging when using accelerate that have been solved by updating the code.
The same code can still be used with a single gpu without any problem.

- There is a logger by accelerate, https://huggingface.co/docs/accelerate/en/package_reference/logging
- And the problem with wandb is likely my restart implementation. I don't remember why I did that.
- https://docs.wandb.ai/guides/integrations/accelerate/

#### Cluster experiments

I have to:

- update the submit script to allow using accelerate
- see how to configure accelerate (if needed)

## Results

### Start point: what is the accuracy of my best models on the train dataset?

| model | dataset    | accuracy | correct_pixels | correct_size | pass_32 | vote_2 |
|-------|------------|----------|----------------|--------------|---------|--------|
| v18   | training   | 30.96%   | 78.82%         | 90.55%       | 58.79%  | 49.29% |
| v19   | training   | 31.68%   | 78.83%         | 90.76%       | 59.13%  | 50.38% |
| v18   | evaluation | 17.05%   | 76.69%         | 90.57%       | 40.88%  | 29.80% |
| v19   | evaluation | 17.91%   | 76.76%         | 90.53%       | 38.25%  | 29.55% |

- Accuracy of the models is very similar, despite v19 being trained for double steps (although it was trained also in the task of generating inputs)
- Train accuracy is clearly higher than evaluation accuracy. This could be because we are using RE-ARC
  for training and also because the training dataset is easier.
- If I want to save time in the task of overfitting, I should probably just train on the task of predicting outputs. Then once
  I have discovered the secret of overfitting I could train again with multiple tasks.
- I believe this metrics shows that either we are using underparametrized models or we are undertraining.
  We should be able to learn to solve all the training tasks.

### Increase model capacity and train duration

TODO: if we want to overfit to the train set, we have to train for longer, use more model capacity or both.

We are exploring two new axis in this experiment: increase the model capacity and train for much longer.

## Conclusion

## Next steps

- Maybe I should train a smaller lora when doing test-time fine-tuning

## TODO

- [ ] Modify the evaluation script to give better names for the output files. This will allow me to evaluate
  both the train and evaluation datasets without overwriting the files.
- [ ] Check if I can do a full fine-tuning instead of LoRA
- [ ] Can I speedup the training by using multiple gpus or unsloth?
