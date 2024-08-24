# Iteration 10. Improve response selection

_24-08-2024_

## Goal

Can I use an LLM to improve the selection of responses over voting?

## Motivation

On the previous iteration we have seen that voting is able to select the correct answer with an accuracy of 30-50%.
If we can find a more accurate method that will make a direct improvement in system accuracy.

We already know that using beam search is able to create better answers, but the implementation is not
efficient and it is much slower than normal generation. My intuition is that we can use the model to estimate
the likelihood of an answer once it has been created. And maybe that can be used to select the right answer.

## Development

### Measuring likelihood of a prompt

I'm going to create a notebook to do a proof of concept of the idea using VLLM. If it works I will
convert it to a script that can replace the current voting script.

I have found that when requiring VLLM to compute the logprobs of the prompt it causes OOM error if the prompt length is not small.
Thus we cannot use that feature. I'm going to try instead to use the logprob of the generated grids.

Links:

- [Github issue: Add efficient interface for evaluating probabilities of fixed prompt-completion pairs](https://github.com/vllm-project/vllm/issues/5234)
- [prompt_logprobs](https://docs.vllm.ai/en/latest/dev/sampling_params.html) – Number of log probabilities to return per prompt token.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
