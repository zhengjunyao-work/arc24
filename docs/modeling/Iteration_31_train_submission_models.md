# Iteration 31. Train new submission models

_25-09-2024_

## Goal

Use all the learnings from the previous iterations to train a new generation of submission models.

Do we get better scores in the leaderboard just for training for longer?

## Motivation

## Development

This is the setup that should give the best results

- Qwen2.5-0.5B-Instruct, I'm using the instruct version because of the problems with the non-instruct version
  on inference.
- Train as long as possible, do continual training (train the same model multiple times). I will first train for 40k steps, then maybe for 80k and continue with the retrainings until we don't see improvements on LB
- All datasets
- 2 tasks
- LoRA seems to generalize better, verify it on the leaderboard

### Resuming the trainings is problematic

I have found that resuming multi-gpu trainings can be problematic. It seems that the script uses the
generator to generate samples until the checkpoint and that takes a lot of time and many times it crashes.
If it happens again I should take a look at the code and configuration of the training.

I have added a parameter `ignore_data_skip` that should speedup the training startup, but then I believe
the problem is related to GPU VRAM. The two trainings that could not be resumed where the full fine-tuning
and the bigger LoRA that used a rank of 128. Those two experiments are the ones that use more VRAM memory.
The training appears to start much faster but fails without a clear error message.

### Weirdly low submission scores

![qwen25-0.5b/8](res/2024-10-07-05-49-57.png)

When doing submissions with my best model `qwen25-0.5b/8` so far some of the submissions score very low (considering
that the best submission got 40 on LB)

I believe that the most plausible explanation for a successful submission with low score is that the
fine-tuning failed, thus the base model was used. I'm going to try to reduce the `max_seq_len`. Inference
should not be affected because I merge the models, so the problem must be on fine-tuning. And the problem
has been manifested only on LoRA with 128 rank, which demands more memory.

I have run the notebook on 10 validation samples and there is no sign of problems: [notebook](https://www.kaggle.com/code/ironbar/single-task-test-time-fine-tuning-for-arc24?scriptVersionId=199978929)

Maybe doing first the trainings with smaller datasets can alleviate the problem: [notebook](https://www.kaggle.com/code/ironbar/single-task-test-time-fine-tuning-for-arc24?scriptVersionId=199994827)

I have downloaded the model and verified that the md5sum of the files is correct. So corruption
of the files does not seem to be the problem.

I have repeated the submission and it scored 37, so maybe there is nothing wrong.

## Results

[Wandb dashboard](https://wandb.ai/guillermobarbadillo/20240925_submission_models?nw=nwuserguillermobarbadillo)

### Leaderboard results

By training the models for longer I have been able to improve the LB score from 36 to 40. Learnings:

- Models trained with LoRA score higher than full fine-tuning. (37 vs 40). My guess is that it is better
  for test-time fine-tuning to fine-tune an already trained LoRA than to fine-tune a new LoRA.
- LoRA with rank 32 scores lower than rank 64 or 128 (37 vs 40)
- Training for 400k steps did not show improvements over training for 200k steps.

### Validation experiments about learning rate and LoRA rank for test-time fine-tuning a new LoRA

![table with results](res/2024-10-13-09-47-28.png)

## Conclusion

On this iteration we have improved the LB score from 36 to 40, reaching the second position in the challenge.

The best results are obtained using LoRA with rank 64 or 128 for 200k steps.

## Next steps

- Single training for 200k steps
- Could I make it work with the non-instruct Qwen version?

## TODO

- [ ] Using small LoRAs might open the door again to use 4 gpus for training. If training for longer is the key, this would be helpful.
- [ ] What is the best learning rate for TTFT? It might change with the lora rank
- [ ] Understand the recent weird low scores on `qwen25-0.5b/8`