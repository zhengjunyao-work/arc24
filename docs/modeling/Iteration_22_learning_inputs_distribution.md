# Iteration 22. Learning the inputs distribution

_10-09-2024_

## Goal

Is it helpful to learn to generate new inputs in addition to learn to solve the tasks?

## Motivation

My intuition is that learning a good representation of the input is the key to solve the challenge. The model will learn the representation by doing tasks that require having a good representation.

One of those tasks is to generate new inputs for a task. The advantage of this approach
is that we don't need to generate new data, we already have it. We just have to make a
better use of it.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ] Refactor the code to allow using different prompts
- [ ] Update fine-tune script to support a more complex configuration for train data (filepath and prompt)
- [ ] Quick experiments to validate implementation
- [ ] Long experiments to see if the model improves
- [ ] Visualize some of the new inputs for the typical first training tasks
