# Iteration 25. Decouple fine-tuning and test-time fine-tuning

_15-09-2024_

## Goal

Can I improve the LB score by decoupling fine-tuning and test-time fine-tuning

## Motivation

My current approach uses LoRA to fine-tune an LLM to do ARC tasks. Then at test-time I fine-tune the same
LoRA to adapt to the test tasks.

It might be the case that using the same lora rank for both stages is not optimal. On the first
training we use around 1k tasks and the training is very long. On the second step we have just 100
tasks and the training is very short. Moreover it seems that fine-tuning the model for each task independently
might be the better option. It is unlikely that we need the same capacity to learn 1k tasks as to learn 1 task.

Moreover I might try to do a full fine-tuning of the model, and in that case I would need to do a different
test-time fine-tuning. That makes worth to investigate the option of decoupling fine-tuning and test-time fine-tuning.

## Development

### LoRA documentation

If I want to adapt to each task, maybe it has more sense to train a small LoRA for each task instead
of retraining the whole LoRA with r=128.

- https://huggingface.co/docs/peft/main/en/developer_guides/lora#pissa
- https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig Maybe `pissa_niter_16` is a good option?

### Local experiment design

I'm going to fine-tune a model on a single task or a few tasks. I will try different LoRa initializations
to see the effect.

```bash
python merge_lora.py --base_model_path /home/gbarbadillo/data/Qwen2-0.5B-Instruct --lora_path /mnt/hdd0/Kaggle/arc24/models/20240910_predict_inputs/10_task-augmentation-and-input-from-inputs-v0_Qwen2-0.5B-Instruct_lr1e-4_r128_2e4steps_10240msl/checkpoint-20000 --output_path /home/gbarbadillo/data/Qwen2-0.5B-arc

jq 'to_entries | .[:5] | from_entries' /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1.json > /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json

python fine-tuning.py \
--model_path=Qwen/Qwen2-0.5B-Instruct \
--adapter_path /mnt/hdd0/Kaggle/arc24/models/20240910_predict_inputs/10_task-augmentation-and-input-from-inputs-v0_Qwen2-0.5B-Instruct_lr1e-4_r128_2e4steps_10240msl/checkpoint-20000 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/01_baseline-from-adapter \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5 \
--learning_rate 4e-5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 32 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/02_LoRA-32-default-initialization \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5 \
--learning_rate 4e-5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 32 \
--lora_weight_initialization pissa \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/03_LoRA-32-pissa \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5 \
--learning_rate 4e-5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 32 \
--lora_weight_initialization default \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/03_LoRA-32-default \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5 \
--learning_rate 4e-5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 8 \
--lora_weight_initialization default \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/03_LoRA-08-default \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5 \
--learning_rate 4e-5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 2 \
--lora_weight_initialization default \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/03_LoRA-02-default \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5 \
--learning_rate 4e-5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 2 \
--lora_weight_initialization default \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/04_LoRA-02-default_lr1e-4 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5 \
--learning_rate 1e-4

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 2 \
--lora_weight_initialization default \
--learning_rate 2e-4 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/04_LoRA-02-default_lr2e-4 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 1 \
--lora_weight_initialization default \
--learning_rate 2e-4 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/04_LoRA-01-default_lr2e-4 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 1 \
--lora_weight_initialization default \
--learning_rate 4e-4 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/04_LoRA-01-default_lr4e-4 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 1 \
--lora_weight_initialization pissa \
--learning_rate 4e-4 \
--warmup_ratio 0.1 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/05_LoRA-01-pissa_lr4e-4 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5

python fine-tuning.py \
--model_path=/home/gbarbadillo/data/Qwen2-0.5B-arc \
--lora_r 1 \
--lora_weight_initialization olora \
--learning_rate 4e-4 \
--warmup_ratio 0.1 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240915_debug_LoRA_initialization/05_LoRA-01-olora_lr4e-4 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1_small.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps=500 \
--logging_steps=10 \
--random_seed=7 \
--batch_size=5
```

### Kaggle experiment design

My idea is to run a few trainings in a few validation tasks with different LoRA configurations. All
the trainings will use batch size 1 by default. I will have a look at the training metrics and also
to the evaluation.

TODO: I would like to visualize the training loss in a plot

## Results

### Weight initialization

I have tried `olora` and `pissa` weight initializations and I haven't observed improvements in training
convergence over the default initialization. In fact I have seen gradient explosions. Thus I recommend
to keep using the default initialization.

[Wandb experiments](https://wandb.ai/guillermobarbadillo/20240915_debug_LoRA_initialization?nw=nwuserguillermobarbadillo)

### Using a new LoRA

I have verified both locally and on Kaggle that I can reach the same training loss with a new LoRA than
simply retraining the base LoRA. Moreover I have been able to reach similar losses using a much smaller LoRA rank (1 vs 32).

I have made 9 submissions with different lora ranks and learning rates without being able to reach
the same LB score as the baseline. Maybe when using a new LoRA we need to train for longer, one way
of achieving that would be to do a common warmup using all the data, then fine-tune for each task
independently.

All the submissions shown below use linear learning rate schedule, batch size 1 and `qwen2-0.5b/19`.

| lora_r | batch size | learning rate | LB score |
|--------|------------|---------------|----------|
| 128    | 1          | 2E-05         | **20**   |
| 1      | 1          | 5E-05         | 16       |
| 1      | 1          | 1E-04         | 17       |
| 1      | 1          | 2E-04         | 16       |
| 32     | 1          | 2E-05         | 17       |
| 32     | 1          | 4E-05         | 18       |
| 32     | 1          | 8E-05         | 14       |
| 128    | 1          | 1E-05         | 19       |
| 128    | 1          | 2E-05         | 18       |
| 128    | 1          | 4E-05         | 16       |

### Using a new LoRA with warmup

TODO: how does the accuracy changes?

## Conclusion

## Next steps

## TODO

- [x] Create a notebook to do experiments: https://www.kaggle.com/code/ironbar/v2-single-task-test-time-fine-tuning-for-arc24?scriptVersionId=196655009
  - [x] Add functionality to visualize training loss, that will allow to compare the different configurations
- [x] Add functionality to train script to select LoRA initialization
- [x] Run local experiments to understand the effect of LoRA initialization
- [ ] Can I get the same or better results when using a new LoRA for test-time fine-tuning?
- [ ] Maybe warming up the adapter by training in all the tasks could be useful. F.e. train with bs=16 for 100 steps.
