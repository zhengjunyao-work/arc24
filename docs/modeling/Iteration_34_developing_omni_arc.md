# Iteration 34. Developing Omni-ARC

_29-09-2024_

## Goal

Implement a first version of Omni-ARC and validate if the approach is promising:

- Can we improve the generalization of the current approach by learning to predict code to do the tasks? Similar to the improvement that we got when learning the inputs distribution.
- Can we use the predicted code to solve evaluation tasks?

## Motivation

## Development

### The right level of abstraction

With the right level of abstraction writing code to solve the training tasks is very easy and fast. I have implemented almost 100 training tasks in less than 2 days. Just with
very basic primitive functions like detect and draw objects is possible to solve a lot of tasks.

### Repeated or very similar tasks

On the training set I have detected some tasks that are exactly the same and other tasks
are just variations of the same task. Maybe the dataset is not as big as I thought.

### First version of Omni-ARC

I have implemented nearly 100 training tasks (25% of the tasks). I believe this is
enough to make a training and see the effect it has on the model.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Update fine-tuning script to support omni-arc dataset
- [ ] Is training speed affected by using omni-arc? I believe generation is fast enough to be done real-time
- [ ] Experiment to see if learning 3 tasks is better than learning two tasks. The baseline learns output-from-examples and input-from-inputs, the new experiment also learns code-from-examples. 10k steps per task.
- [ ] Can we solve some of the evaluation tasks using generated code?
