# Iteration 49. SmolLM2

_04-11-2024_

## Goal

Does SmolLM2 improve over SmolLM?

## Motivation

I have just seen the release of SmolLM2. The biggest change is that the model is prepared to handle
inputs of 8192 tokens (and I believe even longer inputs up to 20k tokens.) The model also improves
the accuracy on some datasets. Thus it is very likely that replacing SmolLM by SmolLM2 will give free
improvements.

## Development

### Tiny modifications to SmolLM2

I'm going to simply increase the `max_position_embedding` from 8192 to 20480, just like I did with `SmolLM`, both
on the `config.json` and `tokenizer_config.json`

### Experiment design

The idea is to run the exact same training experiment I recently did with SmolLM with the new SmolLM2.
Hopefully we will see a faster training loss decrease.

## Results

| model                  | accuracy | pass_n | vote_2 | vote_1 |
|------------------------|----------|--------|--------|--------|
| Qwen2.5-0.5B-Instruct  | 10.32%   | 27.62% | 19.38% | 16.88% |
| SmollLM-135M-Instruct  | 4.20%    | 18.25% | 11.25% | 9.3%   |
| SmollLM2-135M-Instruct | 5.52%    | 20.50% | 12.62% | 9.8%   |

We can see a noticeable improvement when using SmolLM2 over SmolLM, but it is still far from the accuracy
of Qwen2 model.

## Conclusion

SmolLM2 improves over SmolLM, but not enough to be able to compete with Qwen2.5.

## Next steps

## TODO

- [x] Compare the accuracy on evaluation dataset vs SmolLM
