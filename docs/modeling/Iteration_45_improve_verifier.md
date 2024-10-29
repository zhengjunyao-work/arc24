# Iteration 45. Improve the verifier approach

_26-10-2024_

## Goal

On the previous iteration we have seen signs that the verifier approach might work. Let's try to improve that approach.

## Motivation

Having a more accurate method than voting to select predictions could improve the LB score.

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

### Probability of using a wrong prediction for training

The first implementation has hardcoded the probability of using a wrong prediction for training a verifier to 50%.
It uses a balanced dataset of correct and wrong samples.

The problem with this approach is that we have around 1700 correct samples and around 270k wrong predictions.
If we train for 8k steps with a batch size of 32 the model will have seen each correct sample an average of 75 times.
In the other hand the model will have seen on average each wrong prediction 0.5 times.
Maybe it has sense to decrease the frequency of using correct samples for training.

## Results

### Confidence level and verification time

```
without confidence it would take around 2300s

32 verifications max
80%       90%        95%
11440
2960     4168       5152
2448     2816       3296
1712     2480       2864
---------------------------
938s     1087s      1159s
```

23% more time when increasing confidence from 80% to 95%. It is probably worth it.
I can reduce the time to 1100 seconds if I do 4 predictions per round instead of 8.

With this setup I could use up to 128 verifications per prediction in just 2036 seconds.

### Does the verifier work on different models?

| model           | top 1 accuracy | top 2 accuracy |
|-----------------|----------------|----------------|
| voting baseline | 60.00%         | 70.00%         |
| model 1         | 62.90%         | 80.40%         |
| model 2         | 55.90%         | 72.80%         |
| model 3         | 59.00%         | 79.10%         |

We can see that the verifier can work on different models with a similar level of accuracy.
Voting accuracy was almost exactly the same across all the 3 models.
The current method does not seem to be better than voting when selecting the top 1.

### Does training on a bigger dataset improve the accuracy?

Top 1 accuracy table:

| training steps | baseline | more wrong predictions |
|----------------|----------|------------------------|
| 4000           | 55.80%   | 54.60%                 |
| 8000           | 61.70%   | 54.20%                 |
| 16000          | 57.10%   | 62.50%                 |

Top 2 accuracy table:

| training steps | baseline | more wrong predictions |
|----------------|----------|------------------------|
| 4000           | 72.90%   | 72.50%                 |
| 8000           | 77.10%   | 69.60%                 |
| 16000          | 74.60%   | 78.80%                 |

It is unclear if adding more wrong predictions was beneficial.

### Does using task augmentation improve the accuracy?

Top 1 accuracy table:

| training steps | baseline | task augmentation |
|----------------|----------|-------------------|
| 4000           | 54.60%   | 52.50%            |
| 8000           | 54.20%   | 52.10%            |
| 16000          | 62.50%   | 54.2              |

Top 2 accuracy table:

| training steps | baseline | task augmentation |
|----------------|----------|-------------------|
| 4000           | 72.50%   | 72.90%            |
| 8000           | 69.60%   | 72.10%            |
| 16000          | 78.80%   | 76.70%            |

It is unclear if adding task augmentation improves the accuracy. In fact in other experiments the results
are worse.

### Can I achieve perfect accuracy if training on the evaluation set?

| training steps | top_1  | top_2  |
|----------------|--------|--------|
| 4000           | 63.30% | 81.20% |
| 8000           | 62.10% | 89.20% |
| 16000          | 70.40% | 93.80% |
| 32000          | 75%    | 93.30% |

It is surprising that after 32k training steps the model still does not perfectly classify all the
tasks from the evaluation set. After reviewing the failed predictions I have seen that in all the cases
there were ties with other predictions.

On average it would have seen each task 320 times (`16000/4/400*32`), so if a task has 4 samples it would
have seen around 80 times each sample.

### Should I change the probability of training with a correct prediction?

| correct_probability | top_1  | top_2  |
|---------------------|--------|--------|
| 0.1                 | 48.30% | 68.30% |
| 0.2                 | 56.20% | 79.20% |
| 0.3                 | 52.10% | 79.20% |
| 0.5                 | 50.00% | 79.20% |

There is no evidence that suggest that decreasing the probability of using correct predictions gives
higher accuracy.

### Does training for multiple tasks improve the accuracy?

Let's train new models from scratch:

- Add the new verify and select tasks, without task augmentation
- Qwen2.5
- Do the same also for submission, including the evaluation set
- Train for 40k steps with batch size 32.

### Submission results

When using a model to verify the predictions from LLM and 2020 solution I have only achieved a score of 33 when training on the whole ARC, and 30 when training only on the train dataset.

## Conclusion

I have not been able to improve the accuracy of using a prediction verifier. It is still around 60% for top_1 selection and 80% for top_2 selection. Remember that voting gets 60% and 70%. Thus we only see an improving on top_2 prediction.

## Next steps

- Could the verifier benefit from test-time fine-tuning?
- Could I improve the selection of predictions by using selection instead of verifying? I might create a select script by tweaking the verify script.

## TODO

- [x] Create bigger dataset for training
  - [x] Training set
  - [x] Evaluation set
- [x] More data augmentation, allow task augmentation
- [ ] Maybe use an ensemble of models instead of a single model
- [ ] It's likely that a model trained in all the prediction tasks will perform better
- [x] ~Use voting to solve ties~ I won't have voting on 2020 solution.
- [x] I could make more efficient use of compute by using uncertainty and only making verifications for
  the predictions that are not significative different from the top prediction.
- [x] Verify that it works on Kaggle
- [ ] Review all new code
- [ ] Experiments
  - [x] Does training on a bigger dataset improve the accuracy? IN PROGRESS
  - [x] Does using task augmentation improve the accuracy? IN PROGRESS
  - [x] Should I change the probability of training with a wrong prediction? IN PROGRESS
  - [ ] Does training for multiple tasks improve the accuracy?
  - [ ] Train new submission models
- [x] Measure improvements over voting in other model predictions
- [ ] Maybe the model is not as accurate in the test set as in the evaluation set?
- [x] Why cheating did not get perfect accuracy?
- [x] How many verifications I have to do until it reaches the perfect ranking? 128 verifications does not reach significative differences.
      There are ties that avoid reaching the stop point.