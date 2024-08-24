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

## Results

## Conclusion

## Next steps

## TODO

- [ ]
