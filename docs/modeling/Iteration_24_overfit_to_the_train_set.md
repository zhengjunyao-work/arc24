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

## Results

### Start point: what is the accuracy of my best models on the train dataset?

### Increase model capacity and train duration



## Conclusion

## Next steps

- Maybe I should train a smaller lora when doing test-time fine-tuning

## TODO

- [ ] Modify the evaluation script to give better names for the output files. This will allow me to evaluate
  both the train and evaluation datasets without overwriting the files.
- [ ] Check if I can do a full fine-tuning instead of LoRA
- [ ] Can I speedup the training by using multiple gpus or unsloth?
