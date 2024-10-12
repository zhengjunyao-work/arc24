# Iteration 37. Optimize code generation

_10-10-2024_

## Goal

We have verified that we can can solve ARC tasks by generating python code. Let's try to understand the
dynamics of this new training and optimize the hyperparameters.

## Motivation

Solving tasks with code works, but can I optimize and improve the accuracy of the model?

## Development

## Results

### How does the training steps affect the accuracy?

![training steps](res/2024-10-12-12-14-40.png)

We don't see a clear relation between the training steps and model accuracy for this version of omni-arc.

### Is it helpful to learn to do other tasks?

The mean pass_n of the experiments that only learn one task is 3.1%, while the experiments that learn
multiple tasks is 4.1%. So clearly in this experiment learning multiple tasks is beneficial.

### Is it helpful to use a temperature different than 0?

There is great uncertainty in the results, so the best way to study the tendency is to compute the
mean value for all the experiments.

![temperature effect](res/2024-10-12-12-10-35.png)

The improvement is not huge, but we get better results on average when using a temperature of 0.7

### Is there any difference between prompts?

| prompt_version | experiment 1 pass_n | experiment 2 pass_n |
|----------------|---------------------|---------------------|
| 0              | 0.035               | 0.0475              |
| 1              | 0.0425              | 0.0425              |
| 2              | 0.0325              | 0.0425              |

There isn't a clear winner.

### What if I train on omni-arc just to create the output grid?

| experiment           | pass_n | vote_2 | vote_1 |
|----------------------|--------|--------|--------|
| code-from-examples   | 5.00%  | 5.00%  | 5.00%  |
| output-from-examples | 9.25%  | 7.50%  | 6.13%  |

In the best case we solve 5% of the evaluation tasks with the code approach. If we train on the same
data but we predict the grids directly we can get almost double `pass_n`, but `vote_1` is much closer.

So maybe current performance is good for the amount and quality of data we have.

## Conclusion

## Next steps

- Try with bigger models. If test-time fine-tuning is not necessary we might benefit from using bigger or coding models. F.e.
  - https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
  - https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Improve the omni-arc dataset:
  - Add more tasks to increase coverage
  - Add more training inputs to have more variability (can I reuse re-arc for this?)
  - Add task variations
  - Add task to learn to use the primitives

## TODO

- [x] How does the training steps affect the accuracy? -> Run trainings with different training lenght, just using code data
- [ ] What is the best prompt? Is there any difference?
- [x] Is it helpful to learn to do other tasks?
- [ ] Do the results improve if we do test-time fine-tuning?
- [x] What if I train on omniarc just on the default task?
