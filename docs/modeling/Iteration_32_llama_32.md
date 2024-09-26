# Iteration 32. Llama 3.2

_26-09-2024_

## Goal

Try the new Llama 3.2 1B model

## Motivation

Meta has released a new Llama 3.2 1B model. The size is twice the size of Qwen-0.5B but I believe
it is worth trying.

## Development

I have to explore what is the best way to encode the grids for this Llama model.

- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Tokenizer analysis

Exploring how the tokenizer works is crucial to select a good grid encoding. We want to use an
encoding that uses different symbols for each cell in the grid and at the same time that it is able
to represent the grid with the minimum number of tokens possible.

### Local trainings

<details>
  <summary>Click to see bash commands</summary>

```bash
python fine-tuning.py \
--model_path /home/gbarbadillo/data/Llama-3.2-1B-Instruct \
--lora_r 32 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20240926_debug_Llama32/01_baseline \
--max_steps 500 \
--logging_steps 10 \
--batch_size 16 \
--verbose \
--learning_rate 1e-4
```

</details>

## Results

## Conclusion

## Next steps

## TODO

- [ ] Read about the new release. https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- [ ] What is the best way to encode the grids? Check the tokenizer. Maybe create a notebook to do this inspection, it will be useful in the future if more small models are released.
- [ ] What is the optimal learning rate?
- [ ] How is the training and inference speed compared to Qwen2.5-0.5B? Do I have to reduce the number of predictions or the duration of the test-time fine-tuning?