# Iteration 16. Plan next steps

_02-09-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

## Motivation

## Development

The current LLM approach that fine-tunes a LLM on ARC tasks and uses test-time fine-tuning is likely
to scale if I create more training data. Currently we are only using 700 tasks for training. ¿How would
the accuracy improve if I use more training data?
It would be great to have an estimation of the influence of the number of training tasks to take the decision.

I could go task by task from the train set and create by hand similar but different tasks.

Another way would be to think about the priors and build programatic tasks to learn the priors. Ideally
the format would be the same as in the tasks to enable the better knowledge transfer.

Adding some comments to the tasks could enable to create better representations and thus better generalization.
Is there a way to create comments that would withstand the data augmentations?

In the other hand there is the Ryan Greenblat approach. He used a much more powerful model and had to
generate many programs, but I have the power of fine-tuning that he didn't had.
This approach has the advantage of being able to use the python environment as a verifier of the solutions.
The models can refine the solutions given the outputs.
However I have doubts regarding the number of tokens that would be needed to generate the solution. Also
the context lenght needed for training such a model.
If I could come with a method that iteratively creates more training data this could be a much
powerful solution than the LLM approach.

But in essence both solutions need to build a good representation of the problem internally. That is only
achieved by training in a big number of examples. It is possible that the same model could do both tasks.

A stronger model, trained on more tasks, will likely benefit more from test-time fine-tuning because
it has a better initial problem representation.

The writing code approach seems more appealing and general, but it will involve more changes and I believe I should try at least a few weeks the strategy
of creating synthetic data.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
