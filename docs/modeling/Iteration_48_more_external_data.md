# Iteration 48. More External data

_03-11-2024_

## Goal

Can we improve the LB score by using more external data?

## Motivation

The paper [Combining Induction and Transduction for Abstract Reasoning](https://www.cs.cornell.edu/~ellisk/documents/arc_induction_vs_transduction.pdf)
along with a [400k tasks dataset](https://huggingface.co/collections/barc0/synthetic-arc-dataset-6725aa6031376d3bacc34f76) has
been just published.

It is very likely that training my models in this extra data could result on improved accuracy, so I have
to do it and do it fast.

## Development

### Download datasets

```bash
# with git
git lfs install
git clone git@hf.co:datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
git clone git@hf.co:datasets/barc0/100k-gpt4-description-gpt4omini-code_generated_problems
git clone git@hf.co:datasets/barc0/100k-gpt4omini-description-gpt4omini-code_generated_problems
# with wget
wget https://huggingface.co/datasets/barc0/100k-gpt4-description-gpt4omini-code_generated_problems/resolve/main/100k-gpt4-description-gpt4omini-code_generated_problems.jsonl?download=true -O 100k-gpt4-description-gpt4omini-code_generated_problems.jsonl
wget https://huggingface.co/datasets/barc0/100k-gpt4omini-description-gpt4omini-code_generated_problems/resolve/main/100k_gpt4o-mini_generated_problems.jsonl?download=true -O 100k_gpt4o-mini_generated_problems.jsonl
wget https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/resolve/main/data_100k.jsonl?download=true -O data_100k.jsonl
wget https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/resolve/main/data_suggestfunction_100k.jsonl?download=true -O data_suggestfunction_100k.jsonl
```

### Barc dataset object

Since I have seen that the dataset uses a new format, I'm going to create dataset to handle all the
particularities and just expose a sample task method.

### Local experiments to validate implementation

Let's verify that we can train with the BARC datasets:

<details>
  <summary>Click to see bash commands</summary>

```bash
# mixed use of datasets
python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B-Instruct \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241103_debug_BARC/01_Qwen2.5-0.5B-Instruct \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json output-from-examples-v1 \
--train_datasets barc-400-10-/mnt/hdd0/Kaggle/arc24/data/barc/100k-gpt4-description-gpt4omini-code_generated_problems/100k-gpt4-description-gpt4omini-code_generated_problems.jsonl output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--device_map None \
--lora_r 32 \
--max_steps 10 \
--logging_steps 1 \
--eval_steps 200 \
--batch_size 16 \
--learning_rate 1e-4 \
--max_seq_len 4096 \
--no-resume_from_checkpoint \
--random_seed 7 \
--verbose

# just use barc
python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B-Instruct \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241103_debug_BARC/02_Qwen2.5-0.5B-Instruct \
--train_datasets barc-400-10-/mnt/hdd0/Kaggle/arc24/data/barc/100k-gpt4-description-gpt4omini-code_generated_problems/100k-gpt4-description-gpt4omini-code_generated_problems.jsonl output-from-examples-v1 \
--val_dataset /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
--device_map None \
--lora_r 32 \
--max_steps 10 \
--logging_steps 1 \
--eval_steps 200 \
--batch_size 16 \
--learning_rate 1e-4 \
--max_seq_len 4096 \
--no-resume_from_checkpoint \
--random_seed 7 \
--verbose


```

</details>

### Experiment design

The goal of the experimentation is to see if the Barc datasets are useful, and how much
weight should I give to them when training along other datasets. I'm going to train
for just 40k steps and just in the task of predicting the output from the examples to
simplify the experiment.

## Results

## Conclusion

## Next steps

- Train a model to generate code (for next years edition)

## TODO

- [x] Explore the dataset
- [x] How can I train on this new dataset? It is much bigger than the other datasets
- [ ] Does it improve the evaluation accuracy?
- [ ] Does it improve the LB score?
