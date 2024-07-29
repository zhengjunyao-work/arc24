# Iteration 3. Fine-tune on ARC tasks

_29-07-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Let's fine-tune an LLM on ARC tasks and see if:

1. Can I learn the train and eval tasks?
2. Does learning the train or eval tasks improves the accuracy of the model on the other dataset?
3. Does training on train/eval tasks improves the results on test dataset?
4. Does it help to start from a model that learned to count?

## Motivation

On the previous iteration I have fine-tuned a Phi-3 model to learn to count. However it seemed that
the capacity to solve ARC tasks was worsened due to that fine-tuning. I still believe that learning
core knowledge priors is important, but maybe we have to do a multi-step learning process: first learn
the priors and second learn to solve the training tasks.

## Development

### Train script

I'm going to copy and adapt the script that was used to teach the models to count. It's a little bit
dirty but that will allow to start training quickly. Later I could think about refactoring a single
training script.

I'm going to apply rotations and flips to augment the train samples by x8. I also believe I could
swap some of the train samples by the test sample to increase the dataset by an additionally x4 (estimated)
Thus in the best case I will take the 400 train samples and get 12800.

I have concerns about the memory usage. When training to learn to count the number of tokens was below 1k, but here it might grow to 8k.

## Results

## Conclusion

## Next steps

- Could I frame the problem as a 2 player game where the first player needs to describe in text the
  transformation and the second player needs to implement it given the text description and the input?

## TODO

- [ ]
