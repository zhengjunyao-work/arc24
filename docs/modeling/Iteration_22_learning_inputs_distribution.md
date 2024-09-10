# Iteration 22. Learning the inputs distribution

_10-09-2024_

## Goal

Is it helpful to learn to generate new inputs in addition to learn to solve the tasks?

## Motivation

My intuition is that learning a good representation of the input is the key to solve the challenge. The model will learn the representation by doing tasks that require having a good representation.

One of those tasks is to generate new inputs for a task. The advantage of this approach
is that we don't need to generate new data, we already have it. We just have to make a
better use of it.

## Development

```bash
reset; rm -r /mnt/hdd0/Kaggle/arc24/evaluations/20240907_more_data_augmentation/04_100-augmentation-1110_Qwen2-0.5B-Instruct_lr1e-4_r32_12e3steps_10240msl/checkpoint-12000/inference_x008*; python easy_inference_and_evaluation.py /mnt/hdd0/Kaggle/arc24/models/20240907_more_data_augmentation/04_100-augmentation-1110_Qwen2-0.5B-Instruct_lr1e-4_r32_12e3steps_10240msl/checkpoint-12000 --predictions_per_task 8

# Baseline results
accuracy: 3.2%  correct_pixels: 68.8%   max_correct_pixels: 77.4%       correct_size: 90.1%     any_correct_size: 91.0% pass_n: 9.5%    unanswered: 2.0%
accuracy: 3.8%  correct_pixels: 69.7%   max_correct_pixels: 74.7%       correct_size: 90.8%     any_correct_size: 92.3% pass_n: 7.7%    unanswered: 1.5%

# Fix tiny difference between train and inference
accuracy: 3.3%  correct_pixels: 68.9%   max_correct_pixels: 78.2%       correct_size: 90.2%     any_correct_size: 92.0% pass_n: 10.5%   unanswered: 2.0%
accuracy: 3.8%  correct_pixels: 69.7%   max_correct_pixels: 74.5%       correct_size: 90.8%     any_correct_size: 92.3% pass_n: 7.7%    unanswered: 1.5%
```

## Results

## Conclusion

## Next steps

## TODO

- [x] Refactor prompt code to remove duplications, verify that inference results do not change.
- [ ] Refactor the code to allow using different prompts
- [ ] Update fine-tune script to support a more complex configuration for train data (filepath and prompt)
- [ ] Quick experiments to validate implementation
- [ ] Long experiments to see if the model improves
- [ ] Visualize some of the new inputs for the typical first training tasks
