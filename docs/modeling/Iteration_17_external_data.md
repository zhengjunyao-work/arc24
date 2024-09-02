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

## Results

## Conclusion

## Next steps

## TODO

- [ ] Create a dataset with reverse ARC tasks. All tasks that are invertible could be reversed and used for training.
