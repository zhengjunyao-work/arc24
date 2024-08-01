# Iteration 3. Fine-tune on ARC tasks

_29-07-2024_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Let's fine-tune an LLM on ARC tasks and see if:

1. Can I learn/overfit the train and eval tasks?
2. Does learning the train or eval tasks improves the accuracy of the model on the other dataset?
3. Does training on train/eval tasks improves the results on test dataset?
4. Does it help to start from a model that learned to count?

## Motivation

On the previous iteration I have fine-tuned a Phi-3 model to learn to count. However it seemed that
the capacity to solve ARC tasks was worsened due to that fine-tuning. I still believe that learning
core knowledge priors is important, but maybe we have to do a multi-step learning process: first learn
the priors and second learn to solve the training tasks.

## Development

### Iterable dataset

I have tried implementing an Iterable dataset for the train dataset, which would be more memory efficient
and training will start faster. However it seems that `SFTTrainerSFTTrainer` is not ready for it.

### Train script

I'm going to copy and adapt the script that was used to teach the models to count. It's a little bit
dirty but that will allow to start training quickly. Later I could think about refactoring a single
training script.

I'm going to apply rotations and flips to augment the train samples by x8. I also believe I could
swap some of the train samples by the test sample to increase the dataset by an additionally x4 (estimated)
Thus in the best case I will take the 400 train samples and get 12800.

I have concerns about the memory usage. When training to learn to count the number of tokens was below 1k, but here it might grow to 8k.

TODO: color swap (does it have sense?) or to remap the colors on each task

### Going to the cloud

#### AWS

P5 instances have 8xH100 GPUs and P4 instances have 8xA100 GPUs. There does not seem to be an option
with a smaller number of GPUs.

#### Google Cloud

Google cloud allows to create machines with [1 or more A100 GPUs](https://cloud.google.com/compute/docs/gpus#a100-gpus), f.e. `a2-highgpu-1g`, `a2-ultragpu-1g`, `a2-highgpu-2g`... Ultra machines have 80GB of GPU memory, the others have 40GB.

When it comes to [H100 GPUs](https://cloud.google.com/compute/docs/gpus#h100-gpus) we have to use 8, there are no smaller options.

I don't see any other available option in Google Cloud with 40GB or more.

#### [Vast.ai](https://vast.ai/)

The prices here are also much better than in Google Cloud.

#### [Lambdalabs](https://lambdalabs.com/service/gpu-cloud#pricing)

After a quick comparison the prices on Lambdalabs seem to be much better than Google Cloud. So I'm probably starting here.

#### Veridas cluster

### RE-ARC

I have published a [notebook](https://www.kaggle.com/code/ironbar/generate-training-samples-using-re-arc) to generate training data in the same format as ARC tasks.

## Results

- Starting from the model that was taught to count is not helpful, starting loss is higher and also final.
  This follows the bad results observed when trying to solve arc tasks with that model.
- By training on the train dataset the validation loss is decreased
- Data augmentation is helpful to decrease the validation loss
- With 24GB of gpu memory I can only fit one sample of 4096 tokens
- First evaluations show improvement on the train set, but it only solves 1/10 tasks. More training is needed.
- Overfitting to the train data is not easy, 36 epochs with careful learning rate

## Conclusion

## Next steps

- Could I frame the problem as a 2 player game where the first player needs to describe in text the
  transformation and the second player needs to implement it given the text description and the input?
- I need more computing power
- I could study different active inference techniques on the eval dataset. F.e. n-1 train. Eval loss should be a good proxy to see if the different techniques are useful
- [smollm](https://huggingface.co/blog/smollm)

## TODO

- [ ] Evaluate fine-tuned model on arc tasks
- [ ] Does predicting the grid shape helps?
- [ ] Prepare hodel data
