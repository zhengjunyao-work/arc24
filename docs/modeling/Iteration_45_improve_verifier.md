# Iteration 45. Improve the verifier approach

_26-10-2024_

## Goal

On the previous iteration we have seen signs that the verifier approach might work. Let's try to improve that approach.

## Motivation

## Development

### Create bigger datasets for training

#### Training set

By generating more wrong predictions I have increased the size of the training dataset from 48 to 130MB.
The mean number of wrong predictions per training sample has increased from 54 to 155, and the total
number of wrong predictions has increased from 92k to 267k.s

#### Evaluation set

I have created a first dataset with a mean number of wrong predictions per sample of 163, the file
weights 260MB.

### Add task augmentation to verification task

I have to refactor the code to enable using task augmentation with verification, because currently
it is only prepared for `input` and `output` grids, not for `wrong_prediction` grid.

<details>
  <summary>Click to see bash commands</summary>

```bash
python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 128 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241026_debug_task_augmentation/01_baseline_no_task_augmentation \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 10 \
--logging_steps 1 \
--eval_steps 200 \
--batch_size 16 \
--learning_rate 1e-4 \
--max_seq_len 4096 \
--no-resume_from_checkpoint \
--random_seed 7 \
--verbose

python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 128 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241026_debug_task_augmentation/02_task_augmentation_refactor \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 10 \
--logging_steps 1 \
--eval_steps 200 \
--batch_size 16 \
--learning_rate 1e-4 \
--max_seq_len 4096 \
--no-resume_from_checkpoint \
--random_seed 7 \
--compose_new_task_probability 0.5 \
--verbose

python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 128 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241026_debug_task_augmentation/03_revert_refactor \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 10 \
--logging_steps 1 \
--eval_steps 200 \
--batch_size 16 \
--learning_rate 1e-4 \
--max_seq_len 4096 \
--no-resume_from_checkpoint \
--random_seed 7 \
--compose_new_task_probability 0.5 \
--verbose


python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 128 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241026_debug_task_augmentation/04_verify_no_task_augmentation \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/verifier/training_v0.json verify-output-from-examples-v0 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 10 \
--logging_steps 1 \
--eval_steps 200 \
--batch_size 16 \
--learning_rate 1e-4 \
--max_seq_len 4096 \
--no-resume_from_checkpoint \
--random_seed 7 \
--compose_new_task_probability 0.0 \
--verbose

python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B \
--device_map None \
--lora_r 128 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241026_debug_task_augmentation/05_verify_with_task_augmentation \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/verifier/training_v0.json verify-output-from-examples-v0 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--max_steps 10 \
--logging_steps 1 \
--eval_steps 200 \
--batch_size 16 \
--learning_rate 1e-4 \
--max_seq_len 4096 \
--no-resume_from_checkpoint \
--random_seed 7 \
--compose_new_task_probability 0.5 \
--verbose
```

</details>

### More efficient rank estimation using uncertainty

Currently I'm doing n verifications with all the predictions. F.e. I have seen that 32 verifications
per prediction can be enough to select the best predictions.

This works, but my interest is just to select the best 2 predictions and I'm using a lot of compute
to get the ranking for all the predictions. If I estimate the uncertainty for the verification ratio
of all the predictions I could early discard wrong predictions and just focus on verifying the most
promising predictions.

I also thought of using voting as a way to solve ties, but I won't have voting numbers for the 2020 solutions.
So I should focus on improving the efficiency of estimating the ranking with a verifier model.

## Results

## Conclusion

## Next steps

- Could the verifier benefit from test-time fine-tuning?

## TODO

- [x] Create bigger dataset for training
  - [x] Training set
  - [x] Evaluation set
- [x] More data augmentation, allow task augmentation
- [ ] Maybe use an ensemble of models instead of a single model
- [ ] It's likely that a model trained in all the prediction tasks will perform better
- [x] ~Use voting to solve ties~ I won't have voting on 2020 solution.
- [ ] I could make more efficient use of compute by using uncertainty and only making verifications for
  the predictions that are not significative different from the top prediction.
- [x] Verify that it works on Kaggle
- [ ] Review all new code
- [ ] Experiments
  - [ ] Does training on a bigger dataset improve the accuracy? IN PROGRESS
  - [ ] Does using task augmentation improve the accuracy? IN PROGRESS
  - [ ] Should I change the probability of training with a wrong prediction?
  - [ ] Does training for multiple tasks improve the accuracy?
  - [ ] Train new submission models
