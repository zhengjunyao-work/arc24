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

```bash
python fine-tuning.py \
--train_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7_n-1.json \
--adapter_path /mnt/hdd0/Kaggle/arc24/models/20240826_grid_encoders/04_row-number-and-grid-shape_Qwen2-0.5B-Instruct_lr1e-4_r32_6e3steps/checkpoint-6000/ \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240828_grid_encoders_ttft/01_shape-and-number_Qwen2-0.5B-Instruct_lr1e-5_r32_1e3steps \
--learning_rate 1e-5 \
--max_steps 1000
```
TODO: So far there is no sign of stopping improvement after increasing training duration to 12k steps from previous 6k
TODO: plot of val loss, train loss vs val metrics

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
  - [ ] I'm using the best learning rate?
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
