# Iteration 43. Train a verifier

_21-10-2024_

## Goal

Can we improve the LB score by better selecting the predictions?

## Motivation

Currently I use voting to select the predictions of the model for the submission. On [Iteration 9](Iteration_09_improve_inference.md) we saw that voting can select the
best answer with an accuracy of 30-50%.

If we can use a model to improve the answer selection there could be a lot of room
for improvement, in the best case we might triple the score! (it's not going to happen)

The idea is to train a model that is able to select between two possible answers
for a task. Instead of predicting the whole answer it just has to select the correct one.

## Development

## Results

### Generating wrong predictions

It is surprisingly difficult to generate wrong predictions for the training dataset. That is why
I'm going to train new models that do not use the training dataset for training. We can modify
the temperature of the inference to force the errors, but it also increases the number of non valid predictions.

## Conclusion

## Next steps

## TODO

- [ ] Maybe I can force VLLM to generate different predictions for the same prompt?
- [ ] Train a model without the train dataset to generate wrong predictions
- [ ] Create a dataset that can be used to train a verifier.
  - [ ] How do the wrong answers look like?
  - [ ] It has to be of the train dataset, so I can measure the improvement on the evaluation set.
- [ ] Create prompts to select between answers
- [ ] How to integrate this new task into the training script?
  - [ ] How should I format the dataset?
  - [ ] How to apply data augmentation?
- [ ] Train a model to select answers
- [ ] What is the best way to use the model? There might be some compute intensive way and a faster and approximate one
- [ ] Measure the improvement over voting
- [ ] Can I train a single model to do all the tasks?
