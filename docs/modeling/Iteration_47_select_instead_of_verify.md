# Iteration 47. Select instead of verify

_29-10-2024_

## Goal

Can I improve prediction selection if instead of verifying the correctness I select between pairs of predictions?

## Motivation

We have seen that verifying the predictions has predictive power and is a method that can improve
over voting in some situations. But it is not improving the leaderboard score.

Maybe the problem is that if many predictions are considered correct by the model, there is no
way to separate them just by verifying. However if we compare those predictions directly and ask
the model to select the best one, in that case it is possible that we could discriminate the predictions.

## Development

### Create select predictions script

```
export checkpoint_folder=/mnt/hdd0/Kaggle/arc24/models/20241026_improve_verifiers/01_verify-and-select_lora032-Qwen2-0.5B-Instruct_lr5e-5_bs32_16000steps_2gpus_8192msl/checkpoint-16000
python easy_select_and_evaluation.py "${checkpoint_folder}" \
--dataset-path /mnt/hdd0/Kaggle/arc24/data/new_partitions/arc-agi_all_challenges.json \
--predictions-path  /mnt/hdd0/Kaggle/arc24/debug/01_full-fine-tuning-Qwen2-0.5B-Instruct_lr5e-5_8e4steps_2gpus_8192msl_checkpoint-80000_inference_evaluation_x032_just_correct_tasks.json \
--verifications-per-prediction 4
```

## Results

### Analyze submission

On this [notebook](https://www.kaggle.com/code/ironbar/test-2020-solution-script?scriptVersionId=192932864) the 2020 solution
is used with just 1 attempt (the 2 attempt is empty) and scores 24. With [2 attempts](https://www.kaggle.com/code/ironbar/test-2020-solution-script?scriptVersionId=192884747)
it scores 26.

### Analyze verifications

I have studied the failed verifications and found that 36% of them fail with ties. If I can use
the select mode instead to fix that I might have a better prediction selection mode.

### First results with all vs all comparisons

I have created a first implementation that does all vs all comparisons for all the predictions from a task.

| n_rounds | top_1 accuracy | top_2 accuracy | runtime (s) |
|----------|----------------|----------------|-------------|
| 1        | 59.0%          | 75.4%          | 895         |
| 2        | 61.9%          | 79.5%          | 1773        |
| 4        | 69.4%          | 81.7%          | 3655        |
| 8        | 74.3%          | 79.5%          | 7226        |
| 16       | 70.5%          | 83.6%          | 14371       |

These results are very promising because we have improved the top_1 accuracy from ~60% to 74% and
the top_2 accuracy from ~80% to 83%. Thus it seems that doing comparisons between the predictions
is a more accurate method than verifying the predictions.

The drawback is that this implementation is very slow. We are doing all versus all comparisons and
that gives the same compute to the best prediction and to the last. Since we are just interested
in the top_2 predictions I should be able to optimize the runtime by a good margin while preserving
or even improving the accuracy.

## Conclusion

TODO: comparison table of the accuracy of voting, verification and selection.

## Next steps

- If I define some correctness metric over the predictions, that could open the door to a much more
  training dataset that won't be using the correct prediction over an over. It is unclear if this
  would work better.
- Test time fine-tuning could be use to train a verifier, if wrong predictions are created for the
  the test data.

## TODO

- [x] I need more information to understand the failures of verifying. I need to save the results of the verifications for
  deeper analysis.
- [x] Create a first script to do prediction selection with comparisons. First do an all vs all comparison.
- [ ] Update the script to be more efficient and do more selective comparisons, using maybe trueskill.
