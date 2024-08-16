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
- The default empty response is `[[0, 0], [0, 0]]`
- In my opinion it would be more readable if I compute the solution for icecuber with one script, solution with other approaches in other script, and have another script to combine the solutions.
- It seems that icecuber sometimes leaves empty answers `[]`, it happened 8 times on the train set, always on the 2º attempt.
  By looking at the old submission it seems that it can generate even more solutions than 2.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Refactor 2020 solution code and add arguments, there should only be a source of truth for the paths
- [ ] Understand how the solution is generated and use that to my advantage when ensembling
- [ ] Evaluate the solutions for train and evaluation
- [ ] Better analysis of the generated solutions and see what is the best way to ensemble