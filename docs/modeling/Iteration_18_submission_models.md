# Iteration 18. Train models for submission

_03-09-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Train a set of models with the submission in mind. This implies using all the available data for training and skipping the validation step.

## Motivation

It is likely that training in all the available data would bring small improvements on leaderboard.
Solving an extra problem could be the difference between winning a prize or not.

## Development

12k training steps could be a good duration when using the original ARC dataset.
It is likely that when adding new classes I could train for longer and get better results.

Initially I will be using the whole ARC dataset. That way I will have an additional 100 evaluation
tasks that could boost the leaderboard score. I will be adding new data once I verify that it is beneficial.

Using a higher lora rank than 32 might give better results, but I still have to verify it.

Use the bigger `max_seq_len` possible, because there will be long problems in the test set. I have found that I can use
a `max_seq_len` of 10240 without any problem, and very little slowdown because each prompt is processed independently, so they are not padded.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Do we see improvements in LB score when increasing the `max_seq_len`?
- [ ] Do we see improvements when increasing LoRA rank?
