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

All the presented results are the mean metrics of doing 64 predictions per task.

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

**Validation loss is useful when it decreases, but when it diverges it is no longer correlated with model accuracy**

### Longer trainings

Since I have found that the validation loss was not a good metric, I'm going to train the models for longer.

TODO: plot of val loss, train loss, val metrics vs number of training steps for both Qwen models

#### Optimal learning rate

I have tried increasing the default 1e-4 learning rate to see if I could get better results, without success.

![optimal learning rate](res/2024-08-31-14-50-58.png)

| model      | learning rate | train loss | val loss | accuracy | correct_pixels | correct_size | pass_n | unanswered |
|------------|---------------|------------|----------|----------|----------------|--------------|--------|------------|
| Qwen2-0.5B | 1.00E-04      | 0.03       | 0.175    | 3.40%    | 67.50%         | 85.00%       | 18.00% | 2.60%      |
| Qwen2-0.5B | 2.00E-04      | 0.0319     | 0.175    | 2.90%    | 65.50%         | 82.70%       | 19.00% | 3.60%      |
| Qwen2-0.5B | 4.00E-04      | 0.043      | 0.152    | 2.80%    | 65.40%         | 83.00%       | 12.50% | 3.10%      |

#### Effect of lora rank

![effect of lora](res/2024-09-01-09-26-44.png)

- Train loss decreases as the lora rank increases, as expected. Given more capacity the loss is reduced more.
- There seems to be a relation between accuracy and lora rank. We get higher accuracy by using higher ranks.
- The relation with the other metrics is unclear

| model      | lora_rank | train loss | val loss | accuracy | correct_pixels | correct_size | pass_n | unanswered |
|------------|-----------|------------|----------|----------|----------------|--------------|--------|------------|
| Qwen2-0.5B | 16        | 0.0385     | 0.175    | 3.10%    | 66.50%         | 85.00%       | 19.50% | 3.00%      |
| Qwen2-0.5B | 32        | 0.0305     | 0.175    | 3.40%    | 67.50%         | 85.00%       | 18.00% | 2.60%      |
| Qwen2-0.5B | 64        | 0.0249     | 0.189    | 3.20%    | 65.40%         | 82.80%       | 15.00% | 4.10%      |
| Qwen2-0.5B | 128       | 0.021      | 0.1805   | 4%       | 67.30%         | 83.90%       | 18.00% | 3.10%      |
| Qwen2-0.5B | 256       | 0.0194     | 0.1856   | 4.50%    | 67.90%         | 84.60%       | 21.00% | 2.90%      |

### Test time fine-tuning

TODO: What is the best configuration? Show again that we cannot trust validation loss. Trying with higher learning rates and constant schedule

Best result so far are obtained with lr=1e-4 and 4k steps. Maybe training for longer will yield better results.
Cyclic learning rates might speedup training.

Constant schedule was worse than linear.

TODO: a plot of validation loss vs other metrics

#### Cosine with restarts learning rate schedule

I have updated the fine-tuning to support cosine with restarts learning rate scheduler.

- https://huggingface.co/docs/transformers/en/main_classes/trainer
- https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/trainer_utils.py#L410
- https://huggingface.co/transformers/v4.2.2/_modules/transformers/optimization.html

Maybe I could use another scheduler directly, that decreases the amplitude of the cosine restart
over the train duration. The experiment with cosine learning rate seems to be increasing the learning rate to a too high value.

- https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
- https://github.com/huggingface/transformers/blob/746104ba6f0514159d58dc2fb09c887d0e9d4863/src/transformers/trainer.py#L1249C22-L1249C34
- https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/adamw.py

4 cycles, 0.707, warmup ratio in the cycle.
It seems I would need to give both the optimizer and scheduler as input to the train function.

Study of how the [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer#trainer) works ([source code](https://github.com/huggingface/transformers/blob/746104ba6f0514159d58dc2fb09c887d0e9d4863/src/transformers/trainer.py#L289)). It has a method `self.create_optimizer_and_scheduler` that calls to `self.create_optimizer` and `self.create_scheduler`. This method is called from `self._inner_training_loop` that is itself called from the `self.train` method.

- `self.train`
  - `self._inner_training_loop`
    - `self.create_optimizer_and_scheduler`
      - `self.create_optimizer`
      - `self.create_scheduler`
        - `optimization.get_scheduler`

I believe the simplest hack is to modify the `self.create_scheduler` function to return the scheduler I want.

## Conclusion

The biggest finding of this iteration is that validation loss is only useful when it decreases, once it starts to diverge it is no longer useful.

## Next steps

- I might have to reconsider the role of lora ranking now that I know that validation loss is not a good proxy.
  Run a series of experiments with different r. Maybe having a higher r could allow for faster ttft.
- Trainings are becoming too long, could I speedup them using libraries such as [unsloth](https://github.com/unslothai/unsloth)?

## TODO

- [x] Does it help to predict the shape of the grid?
- [x] Does it help to add row idx at the start of each line?
- [ ] Is the system prompt helpful?
- [x] Are the pixel symbols relevant? Or could I replace the number for other symbols?
- [x] How useful is the validation loss?
- [ ] Train for longer, is validation loss really useful?
  - [ ] What is the optimal train steps?
  - [x] I'm using the best learning rate?
  - [ ] Can I get better results using a different lora rank?
- [ ] Test time fine-tuning, train with different number of steps
  - [x] 1e-4 is the best learning rate
  - [x] So far the best results are obtained training for longer, I have trained up to 4k steps
  - [ ] Do I get better results if I train for more than 4k steps?
  - [ ] Can the model learn faster using cyclic learning rates?
  - [ ] Does it help to to remove train samples to fit training sequence length? First experiment gives worse results, but not sure if the differences are significative.
  - [ ] Could I train faster by changing the batch size?
- [ ] Do we get improvements in submission?
- [ ] If I make the same submission 3 times, what is the variability of the score? (Using a random seed)
