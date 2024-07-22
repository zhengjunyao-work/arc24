# Iteration 1. Few-shot prompting

_16-07-2024_

## Goal

How far can we go using few-shot prompting? What is the best way to encode the grids for an LLM?

## Motivation

I want to do a quick iteration where I take an LLM (f.e. phi-3) and use few-shot prompting. I will give different solved problems as input and see how well the LLM do both on the validation and the test set.

## Development

All the work has been done on this [notebook](https://www.kaggle.com/code/ironbar/few-shot-prompting-for-arc24).

I have tried using Phi-3 model and few-shot prompting to solve ARC tasks. I have chosen Phi-3 because its context length of 128k tokens allows to give many ARC tasks as few-shot samples.

### VLLM

Using VLLM allows to use a context size of 61k with 2xT4 GPUs. If I use the transformers library directly
I can only uses a context size of 4k.

### Grid encoding

Some tasks require quite a big context. F.e. imagine a grid of 30x30 that has 4 train examples. At least
we will require `30x30x5x2=9000` tokens. Thus I believe that we should try to use the encoding that uses
the least amount of tokens possible. For Phi-3 that is simply to write the numbers without spaces.

## Results

### Zero-shot baseline

On a first step I have tried a very simple baseline where I give input grids to the assistant and
the assistant replies with the output for each grid. This is done with all the train samples until
we give the test input and use the response of the model as the prediction.
In addition I also use data augmentations (flips and rotations) to make up to two predictions for each task.
The data augmentation is also useful because sometimes the prediction of the model is invalid, so we have to make multiple predictions to have 2 valid responses.

| train | evaluation | test |
|-------|------------|------|
| 6.40% | 2.50%      | 0%   |

This approach is able to solve some of the train and evaluation task, but it does not solve any of the test tasks.

### Few-shot results

Using samples from the evaluation dataset I have evaluated the effect of using few-shot prompting. In this case
I have changed the prompt style: the user shows input-output pairs to the assistant and then requests the assistant
to predict the output given some input.

| n shots | accuracy | correct_pixels | correct_size | unanswered |
|---------|----------|----------------|--------------|------------|
| 0       | 5.80%    | 55.10%         | 73.50%       | 17.40%     |
| 1       | 4.50%    | 44.80%         | 61.00%       | 23.60%     |
| 2       | 4.80%    | 37.70%         | 54.40%       | 29.80%     |
| 4       | 2.50%    | 22.40%         | 33.10%       | 33.10%     |
| 8       | 2.30%    | 23.10%         | 35.50%       | 36.80%     |

The results show that Phi-3 does not benefit from few-shot prompting with ARC tasks. As we give more
examples the results get worse.

### Add reasoning

I have manually described with text the transformation of some of the evaluation tasks. Then repeat
the few-shot experiment but adding the reasoning before creating the grid.

| uses reasoning | accuracy | correct_pixels | correct_size | unanswered |
|----------------|----------|----------------|--------------|------------|
| No             | 2.50%    | 22.40%         | 33.10%       | 33.10%     |
| Yes            | 1%       | 19%            | 30.70%       | 42.50%     |

The model does not understand the puzzles. The examples and reasoning are not useful

### Different models

Since the best results were obtained for the 0-shot setup, I could try using different models.
I can make submissions without using compute time, so I could see if some of the models is able
to solve some task from the test set.

TODO:

## Conclusion

## Next steps

## TODO

- [ ] What is the best way to encode the grids?
- [ ] Does using reasoning and description of the grids helps?
