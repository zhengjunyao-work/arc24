# Iteration 12. Grid representation

_26-08-2024_

## Goal

Can I improve the accuracy of the models by using a better grid representation and better prompts?

## Motivation

Before going to generate synthetic data to learn the priors I want to know what is the best representation
of the problem for an LLM. LLMs typically deal with 1d data, not 2d data. Thus trying different representations
makes a lot of sense. This can have a big impact on the accuracy of the model.

The grid representation should not use too many tokens, otherwise hardware requirements grow.

## Development

### Fine-tuning script refactor

I need to refactor the fine-tuning script to make it very easy to try new representations and prompts.
Code should be shared between train and evaluation.

My idea will be to run a very short train with a fixed seed, refactor the code and verify that it still works the same way.

## Results

### Experiment with different grid encoders

```
GridCodeBlockEncoder(MinimalGridEncoder())
GridCodeBlockEncoder(RowNumberEncoder(MinimalGridEncoder()))
GridShapeEncoder(MinimalGridEncoder())
GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))
```


## Conclusion

## Next steps

## TODO

- [ ] Does it help to predict the shape of the grid?
- [ ] Does it help to add row idx at the start of each line?
- [ ] Is the system prompt helpful?
- [ ] Are the pixel symbols relevant? Or could I replace the number for other symbols?
