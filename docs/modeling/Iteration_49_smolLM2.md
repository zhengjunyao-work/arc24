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

## Conclusion

## Next steps

## TODO

- [ ] Compare the accuracy on evaluation dataset vs SmolLM
