# Iteration 28. Optimal train duration and capacity

_22-09-2024_

## Goal

What is the optimal train duration and capacity to improve the validation score? And for the leaderboard score?

## Motivation

Previous experiments suggest that training for longer might improve generalization. I want to validate that
on the evaluation set.

## Results

### Optimal train duration for full fine-tune

![metrics vs number of steps](res/2024-09-25-08-32-32.png)

All metrics improve when training for longer, we don't see signs of plateau on this experiment. Continual
training might be a good option for this challenge.

### Does LoRA achieve better generalization?

These experiments are for 40k steps, batch size 16.

| experiment       | accuracy | pass_32 | vote_2 | vote_1 |
|------------------|----------|---------|--------|--------|
| full fine-tuning | 12.25%   | 31.13%  | 22.85% | 18.69% |
| LoRA 32          | 11.10%   | 30.25%  | 22.85% | 19.07% |
| LoRA 128         | 12.73%   | 32.25%  | 22.47% | 19.19% |

There is no evidence that full fine-tuning generalizes better than LoRA. 
Moreover considering that so far using a pretrained LoRA for TTFT gives better results it is likely that training a LoRA will give better results

## Conclusion

We have verified that training for longer improves generalization. No sign of plateau was observed after
training for 80k steps. Thus it's likely that when training to do two tasks we could train further than 160k steps
and still see improvements.

So far there is no evidence that full fine-tuning is better than LoRA. I need to validate this on the leaderboard.

## Next steps

- Train new models for submission: full fine-tuning and also LoRA and compare the results on the LB

## TODO

- [x] Determine the optimal training steps for a full model fine-tune
- [x] Do the same for different LoRAs
- [ ] Does the optimal training steps double when also learning to predict input from inputs?
- [ ] Do the results hold for the leaderboard? How to account for the extra training data?
