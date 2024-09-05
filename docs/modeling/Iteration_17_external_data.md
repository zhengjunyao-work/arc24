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
so I can take informed decisions. To characterize uncertainty on evaluation I have run the same evaluation without fixing the random seed, 64 predictions for each task were made (as I have been doing lately). I did the same
experiment with two different models.

| experiment | accuracy | correct_pixels | correct_size | pass_64 | unanswered | pass_2 |
|------------|----------|----------------|--------------|---------|------------|--------|
| 1          | 2.60%    | 68.83%         | 87.36%       | 14.50%  | 2.27%      | 6.63%  |
| 2          | 2.68%    | 69.30%         | 87.77%       | 14.00%  | 2.17%      | 9.18%  |
| 3          | 2.68%    | 68.82%         | 87.45%       | 15.00%  | 2.22%      | 9.18%  |
| 4          | 2.72%    | 68.92%         | 87.53%       | 17.50%  | 2.18%      | 8.67%  |
| 5          | 2.69%    | 69.17%         | 87.58%       | 19.00%  | 2.23%      | 6.63%  |
| mean       | 2.67%    | 69.01%         | 87.54%       | 16.00%  | 2.22%      | 8.06%  |
| std        | 0.04%    | 0.22%          | 0.15%        | 2.15%   | 0.04%      | 1.32%  |

| experiment | accuracy | correct_pixels | correct_size | pass_64 | unanswered | pass_2 |
|------------|----------|----------------|--------------|---------|------------|--------|
| 1          | 2.57%    | 68.44%         | 86.52%       | 13.00%  | 2.44%      | 7.65%  |
| 2          | 2.70%    | 68.82%         | 86.81%       | 18.00%  | 2.18%      | 6.63%  |
| 3          | 2.40%    | 68.48%         | 86.60%       | 13.50%  | 2.39%      | 7.65%  |
| 4          | 2.56%    | 68.54%         | 86.33%       | 15.50%  | 2.41%      | 6.63%  |
| 5          | 2.50%    | 68.42%         | 86.36%       | 15.50%  | 2.36%      | 7.65%  |
| mean       | 2.55%    | 68.54%         | 86.52%       | 15.10%  | 2.35%      | 7.24%  |
| std        | 0.11%    | 0.16%          | 0.20%        | 1.98%   | 0.10%      | 0.56%  |

- It seems that the evaluation has low uncertainty. The difference between runs of the same model are small.
- The only metric that is noisier is `pass_64` and `pass_2`, as expected because there are only 100 tasks, and it aggregates all the predictions for each task.
- `pass_2` is less noisy, probably because it is the result of voting, and pass_64 aggregates all the tasks and takes the best
- Thus we can trust this metrics, we could measure small differences between models if the training is reliable

### Variability on model training

#### Using new partition (700 train, 100 val)

| experiment | accuracy | correct_pixels | correct_size | pass_64 | unanswered | pass_2 |
|------------|----------|----------------|--------------|---------|------------|--------|
| 1          | 2.69%    | 69.17%         | 87.58%       | 19.00%  | 2.23%      | 6.63%  |
| 2          | 2.77%    | 68.72%         | 86.72%       | 20.50%  | 2.21%      | 8.16%  |
| 3          | 3.55%    | 69.82%         | 89.27%       | 17.50%  | 2.27%      | 12.76% |
| mean       | 3.00%    | 69.24%         | 87.85%       | 19.00%  | 2.24%      | 9.18%  |
| std        | 0.48%    | 0.55%          | 1.30%        | 1.50%   | 0.03%      | 3.19%  |

| experiment | accuracy | correct_pixels | correct_size | pass_64 | unanswered | pass_2 |
|------------|----------|----------------|--------------|---------|------------|--------|
| 1          | 2.98%    | 67.67%         | 85.41%       | 16.50%  | 3.05%      | 1.02%  |
| 2          | 2.50%    | 68.42%         | 86.36%       | 15.50%  | 2.36%      | 7.65%  |
| mean       | 2.74%    | 68.05%         | 85.88%       | 16.00%  | 2.70%      | 4.34%  |
| std        | 0.34%    | 0.52%          | 0.67%        | 0.71%   | 0.49%      | 4.69%  |

The variability due to model training is much bigger. In fact this variability will likely make the results very difficult to compare unless there is a big difference between them

How could I reduce the variability to be able to measure small differences between experiments? I have
to take in mind that train loss does not show significative differences between the runs. Thus it does
not seem to be a problem with training convergence. I believe the problem is related to the differences
between training and validation, it is a generalization issue.

- Increasing the validation set will reduce the variability, I could go back to the original train and validation sets.
  I have already created an iteration to train models for submission, so in this iteration I could focus
  on improving the original validation set. If it works I will use that training configuration and train
  on all the data for submission.
- Training for longer might result in more stable predictions
- Cyclic learning rates might also improve convergence, but it doesn't seem to be the problem here.
- If variability is inevitable, the only solution will be to run multiple trainings and average the results.

#### Using original partition (400 train, 400 eval)

I'm going to run multiple trainings with the original partition and measure the variability of the evaluation.

## Conclusion

## Next steps

## TODO

- [x] Create a dataset with reverse ARC tasks. All tasks that are invertible could be reversed and used for training.
- [x] Implement a new option on training that uses a dataset without test tasks. It has a configuration with the number of train tasks and randomly makes new tasks selecting train and test samples.
- [x] Prepare RE-ARC dataset for the new format
- [ ] Once we have the best configuration, try increasing `max_seq_len`. Some training samples do not fit in 4096
- [x] Are VLLM results deterministic? Why?
- [x] What is the uncertainty in my estimation of accuracy? I need to know which differences are significative and which aren't before drawing conclusions.
