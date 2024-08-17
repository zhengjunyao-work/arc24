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

### Train loss is reproducible, val loss isn't

If I run the same experiment multiple times I get very similar train loss, but validation loss could be different.
This makes harder to observe improvements.

![val loss changes](res/2024-08-17-09-27-31.png)

What are the sources of randomness?

- Data augmentation
- LoRA initialization

Maybe cyclic learning rate schedule might allow to escape from poor local optima, but the problem might
be just a generalization one (because training loss is good).

## Conclusion

## Next steps

- [ ] Unify training scripts
- [ ] Iterable for data augmentation will be much better
- [ ] Load the state of the optimizer when fine-tuning in multiple stages? https://chatgpt.com/c/ce6a4f9c-7a50-4c19-a6f3-83793fe6a11d

## TODO

- [ ] What is the effect of changing the train data? Keep the same train configuration and just change the data.
    - [ ] What if I use the train dataset and n-1 eval dataset?
    - [ ] What if I only do test-time fine-tuning?
    - [ ] Is it helpful to first train with re-arc?
    - [ ] Use other arc-like datasets for training
    - [ ] What is the effect of using more training data? Can I estimate how the loss will decrease if generating more tasks?
- [ ] Check for more datasets on: https://docs.google.com/spreadsheets/d/1fR4cgjY1kNKN_dxiidBQbyT6Gv7_Ko7daKOjlYojwTY/edit?gid=658867951#gid=658867951
- [ ] If I can easily swap train and test on fine-tuning, don't do it when creating the n-1 dataset. That will make configuration easier.