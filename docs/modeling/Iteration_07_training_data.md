# Iteration 7. Training data

_15-08-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Try different configurations of training data and see how the validation loss is affected.

## Motivation

## Development

### Create new train-val partition

On the notebook [005_create_new_data_partititions](../../notebooks/005_create_new_data_partititions.ipynb) I have prepared a function
that given a random seed creates a new train-val partition. It will take 100 random eval tasks for validation and the rest for training.

### Prepare arc-like datasets for training

There are some arc-like datasets that could be useful for fine-tuning an LLM. They are listed [here](../02_Data_Understanding.md#external-data)

## Results

## Conclusion

## Next steps

- [ ] Iterable for data augmentation will be much better

## TODO

- [ ] What is the effect of changing the train data? Keep the same train configuration and just change the data.
    - [ ] How the test accuracy changes if I train in the eval set?
    - [ ] How the test accuracy changes if I train in both the train and eval set?
    - [ ] What if I use the train dataset and n-1 eval dataset?
    - [ ] What if I only do test-time fine-tuning?
    - [ ] Is it helpful to first train with re-arc?
    - [ ] Use other arc-like datasets for training
    - [ ] What is the effect of using more training data? Can I estimate how the loss will decrease if generating more tasks?
- [ ] Check for more datasets on: https://docs.google.com/spreadsheets/d/1fR4cgjY1kNKN_dxiidBQbyT6Gv7_Ko7daKOjlYojwTY/edit?gid=658867951#gid=658867951