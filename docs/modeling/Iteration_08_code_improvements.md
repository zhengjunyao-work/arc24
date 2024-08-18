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

#### Experiments

```python
train_dataset = train_dataset.to_iterable_dataset()
Traceback (most recent call last):
  File "/mnt/hdd0/MEGA/AI/22_Kaggle/arc24/scripts/fine-tuning.py", line 754, in <module>
    trainer = SFTTrainer(
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/huggingface_hub/utils/_deprecation.py", line 101, in inner_f
    return f(*args, **kwargs)
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 373, in __init__
    train_dataset = self._prepare_dataset(
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 519, in _prepare_dataset
    return self._prepare_non_packed_dataloader(
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 587, in _prepare_non_packed_dataloader
    tokenized_dataset = dataset.map(
TypeError: IterableDataset.map() got an unexpected keyword argument 'num_proc'
```

This is the same error as shown in [Feature: IterableDataset support for SFTTrainer #1695](https://github.com/huggingface/trl/issues/1695)

Apparently it is solved if I install directly from github:

```
pip install git+https://github.com/huggingface/trl.git

ifferent run name by setting the `TrainingArguments.run_name` parameter.
  0%|                                                                                                                   | 0/100 [00:00<?, ?it/s]Detected flash_attn version: 2.6.3
  5%|█████▎                                                                                                     | 5/100 [00:18<05:26,  3.44s/it]Traceback (most recent call last):
  File "/mnt/hdd0/MEGA/AI/22_Kaggle/arc24/scripts/fine-tuning.py", line 773, in <module>
    trainer.train()
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 444, in train
    output = super().train(*args, **kwargs)
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/transformers/trainer.py", line 1938, in train
    return inner_training_loop(
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/transformers/trainer.py", line 2236, in _inner_training_loop
    for step, inputs in enumerate(epoch_iterator):
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/accelerate/data_loader.py", line 699, in __iter__
    raise ValueError(
ValueError: Batch does not contain any data (`None`). At the end of all iterable data available before expected stop iteration.
Traceback (most recent call last):
  File "/mnt/hdd0/MEGA/AI/22_Kaggle/arc24/scripts/fine-tuning.py", line 773, in <module>
    trainer.train()
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 444, in train
    output = super().train(*args, **kwargs)
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/transformers/trainer.py", line 1938, in train
    return inner_training_loop(
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/transformers/trainer.py", line 2236, in _inner_training_loop
    for step, inputs in enumerate(epoch_iterator):
  File "/home/gbarbadillo/miniconda3/envs/arc/lib/python3.10/site-packages/accelerate/data_loader.py", line 699, in __iter__
    raise ValueError(
ValueError: Batch does not contain any data (`None`). At the end of all iterable data available before expected stop iteration.
```

It seems that the iterator finished, let's try to create an infinite dataset.

```python
def my_generator(dataset):
    while True:
        for item in iter(dataset):
            yield item
train_dataset = IterableDataset.from_generator(my_generator, gen_kwargs={"dataset": train_dataset})
```

This seems to be working!

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