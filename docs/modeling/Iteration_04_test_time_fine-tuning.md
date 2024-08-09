# Iteration 4. Test time fine-tuning

_2-8-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Explore different ways to do test time fine-tuning and see if we can improve
the results on the evaluation dataset.

## Motivation

So far I have seen that I can fine-tune an LLM such as Phi-3 on ARC train tasks and improve the accuracy
of the model on the evaluation dataset. On an interview the MindsAI team said the following:

> If you train a model on the re-arc dataset you will get like 1% on the test set. But if you apply their
> techniques of active inference the score will increase to 23%

Thus although I know that there is room for improvement in the part of training a model in ARC like tasks,
I believe I should investigate first on test time fine-tuning techniques because they seem to be very
relevant for this challenge.

Thus my idea is to explore different ways to use the development dataset (without the test outputs) and
see if I can improve the validation loss.

## Development



## Results

[Training metrics in wandb](https://wandb.ai/guillermobarbadillo/20240802_test_time_fine-tuning?nw=nwuserguillermobarbadillo)

### First results

| experiment                                    | val loss | val accuracy |
|-----------------------------------------------|----------|--------------|
| baseline                                      | 0.1445   | 6.50%        |
| train with n-1 samples                        | 0.1398   | 7.20%        |
| add geometric augmentations                   | 0.1107   | TODO         |
| add color swap                                | 0.099    | 16.80%       |
| preserve original color when using color swap | -        | -            |
| train with n-2 samples                        | -        | -            |

- Test-time fine-tuning has clearly improved the validation loss and accuracy of the baseline model.
- One surprising finding is that using geometric and color augmentations is helpful. I would have said that training with the exact same data would have been better. F.e. preserving the original color when using color swap did not brought improvements.
- Training with n-2 samples did not improve over training with n-1 samples. This matched my intuition because we are not increasing the number of tasks, we are just removing input information.
- Best trainings are taking in the order of 20h in my PC, which is better than Kaggle hardware. This should be at least 4 times faster to be able to use it at submission. Maybe I can try to increase the learning rate to see if the training is faster.

## Conclusion

## Next steps

## TODO

- [ ] Can I think of additional augmentations?
- [ ] Can I speedup training?
- [ ] Can I do test-time fine-tuning on Kaggle with Phi-3? I have doubts about memory requirements. Create a notebook to validate the idea.