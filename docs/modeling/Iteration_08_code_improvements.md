# Iteration 8. Code improvements

_17-08-2024_

## Goal

Improve the code so I can experiment faster in the future.

## Motivation

I have rushed on the previous weeks and now I have to stop, unify and improve the code before trying new approaches.

## Development

### Iterable dataset

Using an iterable dataset will give more control over the data generation. It will also speedup
the start of the training, because currently all the data augmentation is done previously. Additionally
RAM usage will be lower.

Possibly useful links:

- https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#creating-map-style-datasets-and-iterable-datasets
- https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable#eager-data-processing-and-lazy-data-processing
- https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
- https://huggingface.co/docs/trl/en/sft_trainer#datasets
- [Feature: IterableDataset support for SFTTrainer #1695](https://github.com/huggingface/trl/issues/1695)

> In the SFTTrainer we smartly support datasets.IterableDataset in addition to other style datasets. This is useful if you are using large corpora that you do not want to save all to disk. The data will be tokenized and processed on the fly, even when packing is enabled.



## Results

## Conclusion

## Next steps

## TODO

- [x] Unify the train scripts
- [ ] Unify the evaluation scripts
- [ ] Refactor code such as data augmentation to remove duplications
- [ ] Use an iterable dataset to avoid doing all the augmentations before training. This will create
  a better augmented distribution and give more control over the data.
- [ ] Better control over the prompt templates, I would like to try new approaches in the future
- [x] Implement option to load the optimizer when fine-tuning
- [ ] Check if loading an optimizer is helpful for fine-tuning
- [ ] Problem with wandb rate limits:
  - https://docs.wandb.ai/guides/track/limits#rate-limits
  - https://community.wandb.ai/c/w-b-support/36