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

I have to decrease the python version to 3.10, with 3.11 could not find a compatible distribution of xformers.

```
conda create --name arc-unsloth \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
conda activate arc-unsloth

python -c "import torch; print(torch.cuda.is_available())"


pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install wandb termcolor
```

I have accidentally removed conda when trying to clean my base environment :(

Regenerating the arc original environment:

```
conda create -n arc pytest rope pylint tqdm numpy pandas scikit-learn ipython ipykernel coverage ipywidgets matplotlib python=3.10 -y
conda activate arc
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install vllm
```

### Fine-tuning script

On a first step I'm going to duplicate the fine-tuning script and adapt it to use unsloth. Once I validate
that it works and it's faster I will look for the way of having a single script.

## Results

## Conclusion

## Next steps

## TODO

- [ ] It only supports 1 gpu! https://github.com/unslothai/unsloth/issues/547
