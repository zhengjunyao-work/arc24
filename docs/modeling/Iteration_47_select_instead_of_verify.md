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

```bash
export checkpoint_folder=/mnt/hdd0/Kaggle/arc24/models/20241026_improve_verifiers/01_verify-and-select_lora032-Qwen2-0.5B-Instruct_lr5e-5_bs32_16000steps_2gpus_8192msl/checkpoint-16000
python easy_select_and_evaluation.py "${checkpoint_folder}" \
--dataset-path /mnt/hdd0/Kaggle/arc24/data/new_partitions/arc-agi_all_challenges.json \
--predictions-path  /mnt/hdd0/Kaggle/arc24/debug/01_full-fine-tuning-Qwen2-0.5B-Instruct_lr5e-5_8e4steps_2gpus_8192msl_checkpoint-80000_inference_evaluation_x032_just_correct_tasks.json \
--n-rounds 4
```

### Efficient prediction selection

So far we have seen that selection can achieve better accuracy than verify, but at the cost of a much
longer computation time. We have to improve the efficiency of the selection.

```
I'm looking for ranking and a match making system. These are the specifications:

- n players
- 1v1 matches, draw is not an option.
- I'm only interested in finding the best and second best player, I'm not interested in the other players ranking
- Skill does not change over time, is frozen. Thus the order of the matches is irrelevant
- The ranking and match making system should be as efficient as possible, I want to run the smallest number possible of matches because running a match is expensive
- The method should be robust, there is randomness in the results of the matches. F.e. a double elimination tournament is not robust enough because a player can lost two matches just by chance
```

My idea is to work first with a small subset of tasks, and once I have an implementation that feels good enough evaluate all the tasks.

#### Bradley-terry model

I believe Bradley-terry model is the correct way to build a ranking where we have static
players that do not evolve over time. ELO ranking and Trueskill rely on the history of matches, whereas Bradley-terry does a global optimization using all available matches and does not care about the order of the matches.

My only doubt is how fast the algorithm is, but hopefully it should be very fast since
the number of players is not very big.

- [Bradley–Terry model wikipedia](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- [Uncertainty quantification in the Bradley-Terry-Luce model](https://arxiv.org/abs/2110.03874)
- [Bradley-Terry rating system for Kaggle sim comps](https://www.kaggle.com/code/zaharch/bradley-terry-rating-system-for-kaggle-sim-comps)

#### Successive halving algorithms

The idea is that each player plays `n` matches per round, and only the top half players
pass to the next round. If `n` is big enough is very likely that the top 2 players will
end up being selected.

This method should be able to find the best players while being much more efficient than doing all vs all comparisons.

#### Other links from the search

- [Active Learning for Top-K Rank Aggregation from Noisy Comparisons](http://csuh.kaist.ac.kr/Suh_ICML2017.pdf)
- [Active Ranking using Pairwise Comparisons](https://arxiv.org/abs/1109.3701)
- ChatGPT ideas: [1](https://chatgpt.com/c/67288a50-9950-8012-9c29-1973ef9ef2a4), [2](https://chatgpt.com/c/672889bc-f750-8012-830d-567819dbfb3b), [3](https://chatgpt.com/c/672872e6-d018-8012-a7f7-f11b2ba7d757)

#### Implementation details

```bash
export checkpoint_folder=/mnt/hdd0/Kaggle/arc24/models/20241026_improve_verifiers/01_verify-and-select_lora032-Qwen2-0.5B-Instruct_lr5e-5_bs32_16000steps_2gpus_8192msl/checkpoint-16000
rm /mnt/hdd0/Kaggle/arc24/debug/*debug*.json; python easy_select_and_evaluation.py "${checkpoint_folder}" \
--dataset-path /mnt/hdd0/Kaggle/arc24/data/new_partitions/arc-agi_all_challenges.json \
--predictions-path  /mnt/hdd0/Kaggle/arc24/debug/01_full-fine-tuning-Qwen2-0.5B-Instruct_lr5e-5_8e4steps_2gpus_8192msl_checkpoint-80000_inference_evaluation_x032_just_correct_tasks.json \
--n-rounds 8

#quick command to just run inference
python select_predictions_with_halving.py --model-path /home/gbarbadillo/data/temp_model --output-path /mnt/hdd0/Kaggle/arc24/debug/01_full-fine-tuning-Qwen2-0.5B-Instruct_lr5e-5_8e4steps_2gpus_8192msl_checkpoint-80000_inference_evaluation_x032_just_correct_tasks_m14a9650f_001rounds_debug10_selection.json --dataset-path /mnt/hdd0/Kaggle/arc24/data/new_partitions/arc-agi_all_challenges.json --predictions-path /mnt/hdd0/Kaggle/arc24/debug/01_full-fine-tuning-Qwen2-0.5B-Instruct_lr5e-5_8e4steps_2gpus_8192msl_checkpoint-80000_inference_evaluation_x032_just_correct_tasks.json --n-rounds 10
```

When using the first 10 tasks, the naive approach does 953 comparisons each round with the all vs all setup.

### Check that it can work with empty predictions

One of the experiments that I want to try is to make a better selection of the 2020 solution. I know
that the 2020 solution scores 26 on the leaderboard, however if I use a single attempt I only score 24.
Maybe using the selection model I can keep scoring 26.

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

### Improving the selection of 2020 predictions

Using the evaluation dataset we can estimate if we can use model prediction selection to improve the
selection of the submission predictions.

| solution                                               | top_2 accuracy | top_1 accuracy |
|--------------------------------------------------------|----------------|----------------|
| icecuber                                               | 54.0%          | 48.5%          |
| icecuber + model select predictions                    | 54.0%          | 53.5%          |
| program search                                         | 14.0%          | 14.0%          |
| combined submission with preference for program search | 56.5%          | 49.0%          |
| model select predictions                               | 57.0%          | 56.5%          |
| perfect combination                                    | 57.0%          | 57.0%          |

The table shows that when doing the baseline submission combination we get 49% top_1 accuracy and when
we use the model to select predictions we get 56.5% top_1 accuracy which is an amazing improvement.
This has great potential to improve the LB score.

### Efficient implementation results

TODO: same model on different predictions just like the table I did previously
TODO: try different models
TODO: what if I scale the runtime?

### Submission results

TODO: 2020 solution, combined solution, all in 1.

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
