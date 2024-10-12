# Iteration 40. Try coding models

_12-10-2024_

## Goal

Can we improve the results of coding generation by using models that are specialized in code?

## Motivation

I have the intuition that using a model that has been specialized in coding can give better results
than a simple instruct model.

## Development

Qwen already has coding models, so it would be very convenient if I can use those models:

- https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

### Analyze tokenizers

Both coder models use the same tokenizer as the Qwen non-coder models, so I don't have to use a different
grid encoder. I can run the exact same training with these models. The only difference might be
the VRAM requirements.

### VRAM requirements

When training on A6000 gpus that have 48GB of VRAM I can use a `max_seq_len` of 6144 with the 1.5B model.
If I train the 7B model on 2GPUs the `max_seq_len` has to be 4096 or I get OOM errors.

I have checked the trainings with [Llama-3.1-8B](Iteration_20_bigger_models.md) and I also used a `max_seq_len` of 4096.

If I could have access to a GPU with 80GB of VRAM I could increase the training context length.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Can we beat the baseline 5% accuracy of Qwen2.5-0.5B-Instruct with coder models?
- [ ] Inference and train speed comparison of the models
