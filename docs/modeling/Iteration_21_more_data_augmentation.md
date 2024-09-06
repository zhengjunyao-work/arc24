# Iteration 21. More data augmentation

_06-09-2024_

## Goal

Does using more data augmentation improve the accuracy of the model?

## Motivation

When planning about the next steps I identified the opportunity of using [more data augmentation](Iteration_16_next_steps.md#more-data-augmentation).

The idea is to apply extra transformations independently to the input and/or the output. Previously
the same transformation was applied to the input and output.

I believe this extra augmentation should be applied only on the first training, when fine-tuning on ARC tasks.
This extra augmentation will hopefully induce a better representation in the model. But when we are doing
test-time fine-tuning we want to learn the task at hand, so applying extra augmentation isn't likely to
be helpful (at least when we have a small compute budget).

## Development

## Results

## Conclusion

## Next steps

## TODO

- [x] Implement new data transformations, visualize them on a notebook
- [ ] Add parameters to train configuration to allow to control this new data augmentation
- [ ] Run trainings and see if the accuracy improves
