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

### Try on different predictions

```bash
export checkpoint_folder=/mnt/hdd0/Kaggle/arc24/models/20241026_improve_verifiers/01_verify-and-select_lora032-Qwen2-0.5B-Instruct_lr5e-5_bs32_16000steps_2gpus_8192msl/checkpoint-16000
python easy_select_and_evaluation.py "${checkpoint_folder}" \
--predictions-path  /mnt/hdd0/Kaggle/arc24/debug/01_full-fine-tuning-Qwen2-0.5B-Instruct_lr5e-5_8e4steps_2gpus_8192msl_checkpoint-80000_inference_evaluation_x032_just_correct_tasks.json

python easy_select_and_evaluation.py "${checkpoint_folder}" \
--predictions-path  /mnt/hdd0/Kaggle/arc24/debug/05_LoRA-032-Qwen2-0.5B-Instruct_lr1e-4_4e4steps_2gpus_8192msl_checkpoint-40000_inference_evaluation_x032_just_correct_tasks.json

python easy_select_and_evaluation.py "${checkpoint_folder}" \
--predictions-path  /mnt/hdd0/Kaggle/arc24/debug/05_LoRA-128-Qwen2-0.5B-Instruct_lr5e-5_4e4steps_2gpus_8192msl_checkpoint-40000_inference_evaluation_x032_just_correct_tasks.json
```

## Results

### Analyze submission

On this [notebook](https://www.kaggle.com/code/ironbar/test-2020-solution-script?scriptVersionId=192932864) the 2020 solution
is used with just 1 attempt (the 2 attempt is empty) and scores 24. With [2 attempts](https://www.kaggle.com/code/ironbar/test-2020-solution-script?scriptVersionId=192884747)
it scores 26.

My current submission is using that single attempt 2020 submission that scores 24 and combining it
with the result of voting the predictions generated by the transformer. I could use the model to
better select the predictions from 2020 and from the transformer, or I could combine all the predictions
and select among them. My experience with the verifier model suggests that is better to do the selection
of the submissions before combining the two approaches.

There are 3 strategies to improve the submission:

1. Concatenate all the predictions and select the best 2 predictions for each task
2. Select the best 2 predictions for the 2020 solution and my model predictions and combine the predictions
3. Select only best 2 predictions for my model, and combine with the combination of 2020 solution

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

#### Scale runtime

| n_matches per round | top_1 accuracy | top_2 accuracy | runtime (s) |
|---------------------|----------------|----------------|-------------|
| 8                   | 65.70%         | 80.60%         | 873         |
| 32                  | 65.70%         | 83.20%         | 1101        |
| 32                  | 69.40%         | 79.90%         | 1101        |
| 64                  | 69.80%         | 82.80%         | 1658        |
| 128                 | 70.90%         | 82.10%         | 3064        |
| 256                 | 73.50%         | 82.10%         | 6141        |
| 512                 | 71.30%         | 81.00%         | 12175       |

There is too much uncertainty in the metrics to see a clear trend, but using more compute seems to
lead to more accuracy.

#### Different models

| model                   | training tasks | top 1 accuracy | top 2 accuracy |
|-------------------------|----------------|----------------|----------------|
| Qwen2-0.5B              | 2              | 69.4%          | 79.9%          |
| Qwen2.5-0.5B            | 4              | 70.9%          | 82.1%          |
| NanoLM-0.3B-Instruct-v2 | 4              | 54.5%          | 69.4%          |

The table shows that we can achieve similar results when training a model to do more tasks. It also
shows that smaller models give worse results.

### Can I achieve perfect accuracy if training on the evaluation set?

I have used the model `/mnt/hdd0/Kaggle/arc24/models/20241026_improve_verifiers/04_cheat-with-evaluation-no-task-augmentation_lora032-Qwen2-0.5B-Instruct_lr5e-5_bs32_16000steps_2gpus_8192msl/checkpoint-16000` from the previous iteration and with `max_matches_per_round=32` it achieves top_1 accuracy of 84.3% and top_2 accuracy of 94.8%. When using
a verifier this numbers were 75% and 93.30% so there is a clear improvement here.

If I use the model `/mnt/hdd0/Kaggle/arc24/models/20241028_submission_models/01_lora96-Qwen2.5-0.5B-Instruct_lr1e-4_bs32_100000steps_2gpus_8192msl/checkpoint-100000` the accuracy is even higher: 90.7% and 96.3%.
So maybe that is the model that we should be using for submission. This model was also trained to predict
the outputs and the inputs for the ARC tasks. So Omni-ARC approach seems to be working.

### Submission results

I have created a new [notebook](https://www.kaggle.com/code/ironbar/2020-solution-script-prediction-selection?scriptVersionId=205391943) to see if
I can improve the selection of 2020 solution predictions.

#### 2020 solution

First attempts get scores of 20 and 21.5, which implies a top_1 accuracy of 76% and 82% respectively.
This is better than the accuracy on the evaluation set.

The drawback is that is not as a good as the [proof of concept of the 2020 solution on the evaluation data](#improving-the-selection-of-2020-predictions), which achieved an impressing 99% top_1 accuracy.

Maybe is better to do just use the previous 2020 prediction selection which scored 24.

#### Full system

I have made two submissions that use the model to select the predictions of the test-time fine-tuning model.

| selection method | LB score   |
|------------------|------------|
| voting           | 37, 38, 38 |
| model            | 34, 34     |

The results are clear, voting get's higher accuracy than the selection model.

This contradicts my results on the evaluation dataset. Maybe the predictions generated by a test-time
fine-tuned model are more favorable for voting than the predictions of a frozen model.

## Conclusion

| method       | top 1 accuracy | top 2 accuracy |
|--------------|----------------|----------------|
| voting       | 60.0%          | 70.0%          |
| verification | 59.3%          | 77.4%          |
| selection    | **68.7%**      | **81.0%**      |

Using a model to select predictions is the more accurate method of all. It improves an absolute 10%
on top_1 and top_2 over voting.

We implemented an efficient selection algorithm that uses successive halving to focus on the most
promising selections, and the Bradley-Terry ranking system to sort the predictions.

However submission results do not show improvements over voting. My hypothesis is that the predictions
of the test-time fine-tuned model follow a different distribution than the frozen model. And that
different distribution could improve the accuracy of the voting algorithm.

## Next steps

- Test time fine-tuning could be use to train a verifier, if wrong predictions are created for the
  the test data. However I don't believe I have enough compute to do that. If the small LLMs would have worked that would
  have been a real option.

## TODO

- [x] I need more information to understand the failures of verifying. I need to save the results of the verifications for
  deeper analysis.
- [x] Create a first script to do prediction selection with comparisons. First do an all vs all comparison.
- [x] Update the script to be more efficient and do more selective comparisons, using maybe trueskill.
- [x] How does the efficient method scales with more compute? (even when it is not feasible, just to know).
- [x] Try the models trained on the previous iteration 46.
- [x] Submission results
