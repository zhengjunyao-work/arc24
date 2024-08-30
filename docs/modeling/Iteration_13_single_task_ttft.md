# Iteration 13. Single task test-time fine-tuning

_29-08-2024_

## Goal

Can we improve the accuracy of the model if we fine-tune the model for each task independently?

## Motivation

Jack Cole says the following in the [Machine Learning Street interview](https://youtu.be/jSAT_RuJ_Cg?si=8TuDOeS2ip1YMSyv&t=6452):

> If you train a model on the re-arc dataset you will get like 1% on the test set. But if you apply their
> techniques of active inference the score will increase to 23%

This is not what I'm currently seeing. For example when I improved the inference script to be much
faster and added voting, I get LB scores of 6 and 5 for Qwen2-0.5B and Qwen2-1.5B, when applying test-time fine-tuning the scores were 5 and 7.
Thus my initial score is much better than theirs, but I don't experience that improvement when using
test-time fine-tuning.

On recent experiments with the evaluation dataset I could improve from 9% to 15%. Again not even close to what Jack Cole said.
TODO: this is an ongoing work, maybe using longer trainings or higher learning rates could improve my results.

Maybe the problem is that so far I have been fine-tuning in the whole test set. That might be suboptimal because
the tasks could be contradictory. So maybe it's better to fine-tune for each task independently. Instead of fine-tuning for 1k steps on 100 tasks, fine-tune for 10 steps in each of the 100 tasks.

Other possible explanations:

- It might also be the case that we need a stronger base model, but we leave that for future experiments.
- Maybe my test fine-tuning method is not as good as theirs

## Development

The easier way to test this is to fork the Kaggle notebook and make the following modifications.

1. Decompose the test file into single task files
2. Fine-tune on each of those tasks, generating n saved models
3. Make inference with all of the models, each on its task
4. Concatenate all the predictions on a single file

The drawback of this approach is that the warmup time of fine-tuning and inference will happen 100 times
instead of just one. But I believe there is enough time in the submission to do that.

Another possible problem is that if there is a single task, it might be the case that the training data
is too long for training. I will have to think of how to solve that. Probably the best way is to randomly
get rid of some of the inputs in that case. I could add that option to training. Otherwise the data generator
will be stuck in an infinite loop.
However I have found that if task has both inputs and outputs of 30x30, it might be the case that even
with just one train sample we cannot fit the data in the `max_seq_len`. Thus I have to think of how
to deal with those cases.

### Tasks that are too big for training

No prompt smaller than 4096 tokens for task 25094a63
No prompt smaller than 4096 tokens for task f9d67f8b
No prompt smaller than 4096 tokens for task 981571dc
No prompt smaller than 4096 tokens for task b9630600

## Results

## Conclusion

## Next steps

## TODO

- [x] Add the option to the train script to remove train samples to fit training sequence length
- [ ] I have found that sometimes a task would not fit into max seq length. How to detect that case and what to do?
- [ ] Measure the effect of using the feature above in a normal training
- [ ] Optimize the parametrization of single task ttft (stttft) (learning rate and steps) Does it improve over the baseline?
