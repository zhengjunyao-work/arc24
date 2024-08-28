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

### Finding a set of symbols to replace the numbers

Using the Llama and Qwen tokenizers I have been able to find a set of symbols that are unique (do not form part of other words in the vocabulary)
Using this symbols the model should receive a representation that is equivalent to the current numbers one.
But maybe the model can work better with that set of symbols.

```
selection = ['ñ', 'ò', '÷', 'û', 'ą', 'ć', 'ď', 'ę', 'Ě', 'Ğ']
```

### Create a simple evaluation script

Currently I have a notebook to merge weights of the model and lora, a script to do inference and a notebook
to evaluate. That works if I only have to evaluate a single model, but does not scale to evaluating many models.

Thus I have to create either a script or a notebook that simply takes the path of the checkpoint
that I want to evaluate, and does all the job.

## Results

### Experiment with different grid encoders

[Wandb](https://wandb.ai/guillermobarbadillo/20240826_grid_encoders?nw=nwuserguillermobarbadillo)

| row numbers | grid shape  | other symbols | accuracy | correct_pixels | correct_size | unanswered |
|-------------|-------------|---------------|----------|----------------|--------------|------------|
|             |             |               | 2.0%     | 58.0%          | 74.0%        | **2.2%**   |
| x           |             |               | 2.3%     | 63.9%          | 81.5%        | 3.4%       |
|             | x           |               | **2.8%** | 62.3%          | 79.8%        | **2.8%**   |
| x           | x           |               | **2.8%** | **66.3%**      | **84.2%**    | **2.8%**   |
|             |             | x             | 1.9%     | 59.1%          | 75.5%        | **2.4%**   |
| x           | x           | x             | 2.3%     | **66.7%**      | **84.9%**    | 4.1%       |

- Using row numbers and grid shape increase accuracy, correct pixels and correct size
- There is no evidence that using alternative symbols instead of numbers gives better results.

Thus the best encoder configuration for Qwen would be `GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))`.

```
GridCodeBlockEncoder(MinimalGridEncoder())
GridCodeBlockEncoder(RowNumberEncoder(MinimalGridEncoder()))
GridShapeEncoder(MinimalGridEncoder())
GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))

GridCodeBlockEncoder(ReplaceNumberEncoder(MinimalGridEncoder()))
GridShapeEncoder(RowNumberEncoder(ReplaceNumberEncoder(MinimalGridEncoder())))
```

### Validation loss is not a perfect proxy

I have evaluated multiple checkpoints of a training, and the following plots show how the metrics change during the training.

![](res/2024-08-28-17-31-07.png)

We see an almost monotonic improvement during training, however the validation loss shows a different story.

![](res/2024-08-28-17-31-58.png)

Thus I should probably evaluate the last and the best checkpoint, and launch longer trainings because there might be room for improvement.

### Longer trainings

I'm going to train for 12k steps both Qwen models.

TODO:

## Conclusion

## Next steps

## TODO

- [x] Does it help to predict the shape of the grid?
- [x] Does it help to add row idx at the start of each line?
- [ ] Is the system prompt helpful?
- [x] Are the pixel symbols relevant? Or could I replace the number for other symbols?
- [ ] How useful is the validation loss?
- [ ] Train for longer, is validation loss really useful?
- [ ] Test time fine-tuning
- [ ] Do we get improvements in submission?
