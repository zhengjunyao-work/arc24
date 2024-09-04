# Iteration 17. Revisit external training data

_02-09-2024_

## Goal

Add external data to the current training data and see if it improves.

## Motivation

We have made an experiment showing that the model should scale very well when adding new tasks to training.
Additionally we have discovered that validation loss is not a good proxy. Thus we have to [revisit](Iteration_07_training_data.md) the experiments
we did with external data.

## Development

The baseline will be the model trained for 6k steps just on the original ARC tasks. We will run experiments
adding different data and evaluate to see if there is improvement.

### Reverse ARC dataset

Let's loop over all the ARC train tasks and collect the task ids of the tasks that could be reversed (going from the output to the input).

One way to do that would be to use the website and manually copy the ids of the tasks that could be reversed.
However that involves many manual steps. I would rather create a simple jupyter notebook interface
to do that. That could be the base for future work in task generation.

### Is inference deterministic?

I have the feeling that when running inference twice (due to errors or bugs) the results were exactly the same.
That was not my intention, so I want to verify it and fix it.

On a first step I'm going to run inference two times with the same number of predictions. I have verified that the predictions are exactly the same, thus I'm going to check if I'm fixing the random seed. Inference should be deterministic since the temperature is 0, but I'm doing data augmentation which I would like to be random.

VLLM was setting the random seed when creating the LLM, thus I have made a small modification to manually set my seed after that.

### More external datasets

[External datasets are listed here](../02_Data_Understanding.md#external-data)

- [ConceptARC](https://neoneye.github.io/arc/?dataset=ConceptARC) looks nice

## Results

### Uncertainty in the evaluation

The evaluation has some uncertainty, and the training process another. I need to characterize both of them
so I can take informed decisions.

## Conclusion

## Next steps

## TODO

- [x] Create a dataset with reverse ARC tasks. All tasks that are invertible could be reversed and used for training.
- [x] Implement a new option on training that uses a dataset without test tasks. It has a configuration with the number of train tasks and randomly makes new tasks selecting train and test samples.
- [x] Prepare RE-ARC dataset for the new format
- [ ] Once we have the best configuration, try increasing `max_seq_len`. Some training samples do not fit in 4096
- [ ] Are VLLM results deterministic? Why?
- [ ] What is the uncertainty in my estimation of accuracy? I need to know which differences are significative and which aren't before drawing conclusions.
