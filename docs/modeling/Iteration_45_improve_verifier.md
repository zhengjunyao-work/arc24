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

## Results

## Conclusion

## Next steps

- Could the verifier benefit from test-time fine-tuning?

## TODO

- [ ] Create bigger dataset for training
  - [ ] Training set
  - [ ] Evaluation set
- [ ] More data augmentation, allow task augmentation
- [ ] Maybe use an ensemble of models instead of a single model
- [ ] It's likely that a model trained in all the prediction tasks will perform better
- [ ] Use voting to solve ties
- [ ] I could make more efficient use of compute by using uncertainty and only making verifications for
  the predictions that are not significative different from the top prediction.
- [ ] Verify that it works on Kaggle
- [ ] Review all new code
- [ ] Experiments
  - [ ] Does training on a bigger dataset improve the accuracy?
  - [ ] Does using task augmentation improve the accuracy?
  - [ ] Does training for multiple tasks improve the accuracy?
