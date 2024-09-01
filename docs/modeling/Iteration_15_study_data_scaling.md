# Iteration 15. Study how well this method scales with data

_01-09-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

How the model accuracy scales with the number of training tasks?

## Motivation

Before taking a decision about the next steps, I want to know how well the current method scales with
the available training data.

## Development

The idea is to compare trainings that use the same number of steps (same compute) but use different
training data. I'm going to add an option to the fine-tuning script to subsample the train data.

I already have baseline results without subsampling. I'm going to try the following values: `[0.8, 0.6, 0.4, 0.2, 0.1]`

## Results

## Conclusion

## Next steps

## TODO

- [ ]
