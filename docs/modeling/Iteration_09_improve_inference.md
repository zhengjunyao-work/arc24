# Iteration 9. Improve inference

_21-08-2024_

## Goal

Can we improve the accuracy of the LLM by using a more advanced inference?

## Motivation

> MindsAI team uses AIRV (augment, inference, reverse augmentation, and vote)

This simply means that they make multiple inferences for each task and then vote. Voting can be done
at the grid level or we could vote by cell. I have to visualize the predictions to see if we could
increase the accuracy by using some of this techniques.

Another option to try to improve the accuracy is to use beam-search. Instead of using the naive approach of doing greedy generation we could try different options and create an answer that has an overall higher probability.

## Development

It's a good opportunity to refactor the inference code and make it more flexible. I would save all the predictions
in a file and then have a function to take a decision. I will have to create a notebook to visualize the predictions, probably sorted by some metric.

So far I'm only doing geometric augmentations, but I could also do color swaps and change the order of the train samples. That will increase the compute used at inference, but I could probably optimize the inference speed.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
