# Iteration 42. Improve Omni-ARC

_14-10-2024_

## Goal

The idea is to devote at least a week to improve Omni-ARC dataset. At the same time I will run
daily trainings with the updated versions of the dataset and measure progress on the evaluation dataset.

## Motivation

The initial version of Omni-ARC has around 150 training tasks. A model trained on that version is
able to solve just 5% of the evaluation dataset. There should be a lot of room for improvement:

- Add more tasks to increase coverage
- Add more training inputs to have more variability (can I reuse re-arc for this?)
- Add task variations
- Add task to learn to use the primitives

## Development

## Results

### First results

| omni-arc training tasks | training coverage | training pass_8 | evaluation pass_8 |
|-------------------------|-------------------|-----------------|-------------------|
| 150                     | 37.50%            | 35.80%          | 4.50%             |
| 269                     | 67.25%            | 62.60%          | 3.75%             |

First validation results do not show improvements on evaluation after increasing the number of tasks from 100 to 269.
The model is able to solve more training tasks, but its accuracy does not improve on the evaluation set.
These seems like a clear sign of overfitting.

I have visualized the tasks that it does correctly and they are all very similar to the training tasks. Thus
another evidence for overfitting.

Another explanation is that coverage on the evaluation dataset has not increased despite close to doubling
the coverage on the training dataset. But that seems to be very unlikely. I could measure coverage
on the evaluation set by implementing the evaluation tasks.

How could we reduce overfitting and improve generalization?

- Add more input samples. Currently we are just using the original task inputs with some data augmentation.
- Add more tasks. I could create task variations from the original tasks, or create entirely new
  tasks using the omni-arc domain specific language

This two actions should force the model to better learn the relation between examples and code.

## Conclusion

## Next steps

- Many times I do an initial implementation and there is some little detail wrong. I correct the implementation
  and then it's fine. The ideal system should also have this opportunity. However training such a model
  will require a larger context length, and I cannot afford it with the current hardware.
- I have found one task that required looking at the test input to properly implement the function.
  Create a new version of the prompt that uses also the test input.

## TODO

- [ ]
