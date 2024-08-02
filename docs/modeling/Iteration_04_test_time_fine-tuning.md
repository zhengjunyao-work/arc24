# Iteration 4. Test time fine-tuning

_2-8-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Explore different ways to do test time fine-tuning and see if we can improve
the results on the evaluation dataset.

## Motivation

So far I have seen that I can fine-tune an LLM such as Phi-3 on ARC train tasks and improve the accuracy
of the model on the evaluation dataset. On an interview the MindsAI team said the following:

> If you train a model on the re-arc dataset you will get like 1% on the test set. But if you apply their
> techniques of active inference the score will increase to 23%

Thus although I know that there is room for improvement in the part of training a model in ARC like tasks,
I believe I should investigate first on test time fine-tuning techniques because they seem to be very
relevant for this challenge.

Thus my idea is to explore different ways to use the development dataset (without the test outputs) and
see if I can improve the validation loss.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ]
