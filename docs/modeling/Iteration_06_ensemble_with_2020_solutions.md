# Iteration 6. Ensemble with 2020 solution

_14-08-2024_

## Goal

There is a public notebook that scores 26 just using program search. If I can ensemble that with my
current LLM approach there is a good chance to achieve a higher score (hopefully the LLM and program search will be very orthogonal)

## Motivation

Using LLMs and test-time fine-tuning I'm currently scoring 5 on the leaderboard. However there is a [public
notebook](https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions) that scores 26 and runs in 3-4 hours.

There is a good opportunity to create an ensemble. And if I can improve the accuracy of the LLMs the accuracy
of the ensemble will also likely go up.

## Development

### Explaining the notebook

I will write my preliminary understandings of what the notebook is doing (they might change over time when I become more familiar with the code.)

- As far as I understand the basis of the solution is icecuber solution from 2020
- On top of that they try specialized approaches that might work on some tasks
- In my opinion it would be more readable if I compute the solution for icecuber with one script, solution with other approaches in other script, and have another script to combine the solutions. And that is what I have done.
- It seems that icecuber sometimes leaves empty answers `[]`, it happened 8 times on the train set, always on the 2º attempt. Sometimes it generates more than 2 answers. This might be exploited in the future if I can train a model
  to predict what is the correct answer given a series of options. Maybe simply the logloss could be enough.

## Results

### First attempt is clearly better than second for both approaches

The results below show a big drop in accuracy and precision when using the second attempt instead of the first.

#### Icecuber approach

```
# training
attempt_1 recall: 100.0% precision: 46.2% accuracy: 46.2%
attempt_2 recall: 93.5% precision: 6.2% accuracy: 5.8%
# evaluation
attempt_1 recall: 100.0% precision: 34.4% accuracy: 34.4%
attempt_2 recall: 96.9% precision: 4.9% accuracy: 4.8%
```

#### Multiple solvers approach

```
# training
attempt_1 recall: 16.3% precision: 57.4% accuracy: 9.4%
attempt_2 recall: 11.5% precision: 6.2% accuracy: 0.7%
# evaluation
attempt_1 recall: 8.6% precision: 72.2% accuracy: 6.2%
attempt_2 recall: 6.4% precision: 0.0% accuracy: 0.0%
```

### Give preference to multiple solvers approach

As it is shown in the previous section the multiple solvers approach has higher precision on the first attempt,
so it's better to give preference to that approach. Next results show that we achieve higher
accuracy on the first attempt by doing that.

```
# Training results
## preference to multiple solvers
attempt_1 recall: 100.0% precision: 46.4% accuracy: 46.4%
attempt_2 recall: 96.4% precision: 8.5% accuracy: 8.2%
## preference to icecuber
attempt_1 recall: 100.0% precision: 46.2% accuracy: 46.2%
attempt_2 recall: 96.4% precision: 8.7% accuracy: 8.4%

# Evaluation results
## preference to multiple solvers
attempt_1 recall: 100.0% precision: 34.6% accuracy: 34.6%
attempt_2 recall: 97.6% precision: 5.9% accuracy: 5.7%
## preference to icecuber
attempt_1 recall: 100.0% precision: 34.4% accuracy: 34.4%
attempt_2 recall: 97.6% precision: 6.1% accuracy: 6.0%
```

Since we want to ensemble this solution with my LLM approach I should give preference to the multiple
solvers approach to have the best accuracy on the first attempt.

### Submissions

## Conclusion

## Next steps

- Icecuber solution is slow, ideally I would run it on the background while i do all the LLM stuff.

## TODO

- [x] Refactor 2020 solution code and add arguments, there should only be a source of truth for the paths
- [x] Understand how the solution is generated and use that to my advantage when ensembling
- [x] Evaluate the solutions for train and evaluation
- [x] Better analysis of the generated solutions and see what is the best way to ensemble
