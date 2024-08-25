# Iteration 11. Pseudo beam-search

_25-08-2024_

## Goal

Can I improve the accuracy of the predictions by using a pseudo-beam-search?

## Motivation

Beam-search has been probed to generate more accurate responses than greedy decoding.
However it is not efficiently implemented on VLLM.

My idea is to generate n responses for the same prompt and select the one with the highest
logprob. This would be similar to beam-search, but the implementation would be much more efficient.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ] Modify inference script to support this
  - [x] Are the outputs provided by LLM sorted by logprob, or I have to sort them myself? YES THEY ARE ALREADY SORTED
- [ ] How does the inference speed changes when requesting more responses per prompt?
- [ ] Does the accuracy improves?
