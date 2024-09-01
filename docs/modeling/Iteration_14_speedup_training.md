# Iteration 14. Speedup training with unsloth

_01-09-2024_

## Goal

Can we speedup training using [unsloth](https://github.com/unslothai/unsloth)?

## Motivation

On previous iterations I have tried using different learning rate schedules to speedup the training, or changing
the batch size without success. [Unsloth library](https://github.com/unslothai/unsloth) might be an easy way to speedup training.

If I'm able to speedup training I will be able to train faster, but at the same time I could do longer
test-time fine-tuning when making a submission.

Hopefully this will be a very fast iteration.

## Development

### Install unsloth

I'm going to create a new local conda environment following [their instructions](https://github.com/unslothai/unsloth?tab=readme-ov-file#conda-installation)

```
conda create --name unsloth \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### Fine-tuning script

On a first step I'm going to duplicate the fine-tuning script and adapt it to use unsloth. Once I validate
that it works and it's faster I will look for the way of having a single script.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
