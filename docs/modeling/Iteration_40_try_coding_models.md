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

## Results

## Conclusion

## Next steps

## TODO

- [ ]
