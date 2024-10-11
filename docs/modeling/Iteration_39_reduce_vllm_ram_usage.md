# Iteration 39. Reduce VLLM RAM usage

_11-10-2024_

## Goal

If I can reduce the RAM usage from VLLM ensembling would be easier in the Kaggle submission.

## Motivation

Currently I cannot parallelize everything in the submission pipeline because VLLM uses 50% of RAM and the 2020 solution sometimes demands more than 50%.

## Development

I have been playing with VLLM parameters and `swap_space` seems to be the one with the biggest effect on RAM usage. In the documentation it says: 

> CPU swap space size (GiB) per GPU.

```bash
python inference.py --model_path /home/gbarbadillo/data/temp_model --output_filepath /mnt/hdd0/Kaggle/arc24/evaluations/20241007_batch_size/01_bs16_lr5e-5_Qwen2.5-0.5B-Instruct_10000steps_2gpus_8192msl/checkpoint-10000/inference_evaluation_x009.json --predictions_per_task 9 --grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" --dataset_path /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json --prompt_version output-from-examples-v1 --temperature 0.0
```

## Results

| swap_space | RAM usage | inference time (s) |
|------------|-----------|--------------------|
| 4          | 16        | 530                |
| 2          | 9.7       | 563                |
| 1          | 5.5       | 514                |
| 0          | 1.1       | 508                |

We have reached an enormous VRAM decrease without a significant effect on inference time nor in accuracy.
This results were obtained on my PC, I should repeat the experiments on Kaggle.

## Conclusion

## Next steps

## TODO

- [ ]
