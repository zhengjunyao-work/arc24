# Iteration 5. Search for smaller LLMs

_09-08-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Search an LLM that is smaller than Phi-3 and is fast enough to be fine-tuned on Kaggle within the submission time (12 hours)

Can I get a better result than Phi-3 by using a smaller model with test time fine-tuning?

## Motivation

In the previous iteration I have seen that test time fine-tuning works. However when doing a proof of concept
on Kaggle's hardware I have found that to be able to fine-tune Phi-3 with a sequence length of 4096 first I have
to use int4 quantization and second the fine-tuning would take 56 hours for 1k steps.

## Development

### Candidate search

- I could try with previous versions of Phi-3 that I believe were smaller.
- I have also read recently about [smollm](https://huggingface.co/blog/smollm)
- [MobileLLM](https://github.com/facebookresearch/MobileLLM) is cited by smollm, seems to be developed by Facebook but the models do not seem to be available.

However Smollm models have only 2k context length.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Can I run Smollm models with VLLM? The architecture is supported... https://docs.vllm.ai/en/latest/models/supported_models.html
- [ ] Long context fine-tuning https://huggingface.co/blog/wenbopan/long-context-fine-tuning