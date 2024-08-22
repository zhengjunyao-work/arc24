# Iteration 9. Improve inference

_21-08-2024_

## Goal

Can we improve the accuracy of the LLM by using a more advanced inference?

## Motivation

> MindsAI team uses AIRV (augment, inference, reverse augmentation, and vote)

This simply means that they make multiple inferences for each task and then vote. Voting can be done
at the grid level or we could vote by cell. I have to visualize the predictions to see if we could
increase the accuracy by using some of this techniques.

Another option to try to improve the accuracy is to use beam-search. Instead of using the naive approach of doing greedy generation we could try different options and create an answer that has an overall higher probability.

## Development

It's a good opportunity to refactor the inference code and make it more flexible. I would save all the predictions
in a file and then have a function to take a decision. I will have to create a notebook to visualize the predictions, probably sorted by some metric.

So far I'm only doing geometric augmentations, but I could also do color swaps and change the order of the train samples. That will increase the compute used at inference, but I could probably optimize the inference speed.

### Beam search

- [What is Beam Search? Explaining The Beam Search Algorithm](https://www.width.ai/post/what-is-beam-search)
- [VLLM sampling params](https://docs.vllm.ai/en/latest/dev/sampling_params.html)

> **use_beam_search** – Whether to use beam search instead of sampling.  
> **best_of** – Number of output sequences that are generated from the prompt. From these best_of sequences, the top n sequences are returned. best_of must be greater than or equal to n. This is treated as the beam width when use_beam_search is True. By default, best_of is set to n.

### Update inference script

It takes 13 minutes to do inference in 100 evaluation tasks.
54 minutes to do inference with 8 predictions per task.

## Results

### Beam search speed

```
Evaluation on 5 tasks, either beam search is not improving or I'm not using it correctly.

27s dict(temperature=0.0, max_tokens=1000)

accuracy: 0.0%	correct_pixels: 48.8%	correct_size: 60.0%	unanswered: 10.0

45s dict(temperature=0.0, max_tokens=1000, use_beam_search=True, best_of=3)
accuracy: 0.0%	correct_pixels: 48.7%	correct_size: 60.0%	unanswered: 10.0%

1m34 dict(temperature=0, max_tokens=1000, use_beam_search=True, best_of=10)
accuracy: 0.0%	correct_pixels: 48.7%	correct_size: 60.0%	unanswered: 10.0%

2m39 dict(temperature=0, max_tokens=1000, use_beam_search=True, best_of=20)
accuracy: 0.0%	correct_pixels: 48.7%	correct_size: 60.0%	unanswered: 10.0%

ValueError: temperature must be 0 when using beam search.
```

## Conclusion

## Next steps

## TODO

- [ ] Modify generation script to allow generating an arbitrary number of solutions
- [ ] Create a function to select the solution
- [ ] Create a notebook to understand how beam search works, first using text
- [ ] Can I speedup inference?
