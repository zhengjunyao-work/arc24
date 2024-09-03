# Iteration 17. Revisit external training data

_02-09-2024_

## Goal

Add external data to the current training data and see if it improves.

## Motivation

We have made an experiment showing that the model should scale very well when adding new tasks to training.
Additionally we have discovered that validation loss is not a good proxy. Thus we have to revisit the experiments
we did with external data.

## Development

The baseline will be the model trained for 6k steps just on the original ARC tasks. We will run experiments
adding different data and evaluate to see if there is improvement.

### Reverse ARC dataset

Let's loop over all the ARC train tasks and collect the task ids of the tasks that could be reversed (going from the output to the input).

One way to do that would be to use the website and manually copy the ids of the tasks that could be reversed.
However that involves many manual steps. I would rather create a simple jupyter notebook interface
to do that. That could be the base for future work in task generation.

## Results

## Conclusion

## Next steps

## TODO

- [x] Create a dataset with reverse ARC tasks. All tasks that are invertible could be reversed and used for training.
- [x] Implement a new option on training that uses a dataset without test tasks. It has a configuration with the number of train tasks and randomly makes new tasks selecting train and test samples.
- [x] Prepare RE-ARC dataset for the new format
- [ ] Once we have the best configuration, try increasing `max_seq_len`. Some training samples do not fit in 4096
