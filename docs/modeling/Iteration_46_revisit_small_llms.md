# Iteration 46. Revisit small LLMs

_28-10-2024_

## Goal

Can I train a smaller LLM than Qwen2.5-0.5B to achieve the same accuracy?

## Motivation

I don't have too many ideas that can be implemented in 2 weeks. If I can train a smaller model than
then 0.5B Qwen then I could do a longer test-time fine-tuning during the submission or do more
inference steps. That could translate to higher LB scores.

I encountered two problems on past iterations:

1. Small models typically have a context length of 2k or less
2. Some models don't even have a chat template

## Development

### Available options

It's quite difficult to search for an LLM with a certain configuration. I have found a [leaderboard](https://huggingface.co/datasets/open-llm-leaderboard/contents/viewer/default/train?sort%5Bcolumn%5D=%23Params+%28B%29&sort%5Bdirection%5D=asc&row=512)
that allows to sort by the number of parameters. I have also found [awesome-mobile-llm](https://github.com/stevelaskaridis/awesome-mobile-llm).

| model                                                                                           | parameters (M) | max_position_embeddings | rope_theta | attention heads | has chat-template? |
|-------------------------------------------------------------------------------------------------|----------------|-------------------------|------------|-----------------|--------------------|
| [AMD-Llama-135m](https://huggingface.co/amd/AMD-Llama-135m)                                     | 135            | 2048                    | 1.00E+04   | 12              | FALSE              |
| [HuggingFaceTB/SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct) | 135            | 2048                    | 1.00E+04   | 9               | TRUE               |
| [TinyMistral-248M-Instruct](https://huggingface.co/Locutusque/TinyMistral-248M-Instruct)        | 248            | 32768                   | 1.00E+04   | 32              | FALSE              |
| [OpenELM-270M](https://huggingface.co/apple/OpenELM-270M)                                       | 270            | 2048                    | -          | ?               | FALSE              |
| [NanoLM-365M-Base](https://huggingface.co/Mxode/NanoLM-365M-Base)                               | 365            | 131072                  | 1.00E+06   | 14              | TRUE               |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                      | 500            | 32768                   | 1.00E+06   | 14              | TRUE               |

The SmolLM model has an uneven number of attention heads and VLLM does not support model parallel in that case. However
I might not need to use 2 GPUs for such an small model.

### Adding a chat template

I have noticed that Qwen has the chat template in the [tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json#L198).

It seems that I can simply copy it and assign to the AMD-Llama-135m model. [How do I create a chat template?](https://huggingface.co/docs/transformers/main/en/chat_templating#how-do-i-create-a-chat-template)

### Local trainings to verify it can train

<details>
  <summary>Click to see bash commands</summary>

```bash
python fine-tuning.py \
--model_path /home/gbarbadillo/data/Qwen2.5-0.5B-Instruct \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241028_debug_small_LLMs/01_Qwen2.5-0.5B-Instruct \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json output-from-examples-v1 \
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

python fine-tuning.py \
--model_path /home/gbarbadillo/data/NanoLM-0.3B-Instruct-v2 \
--output_dir /mnt/hdd0/Kaggle/arc24/models/20241028_debug_small_LLMs/02_NanoLM-0.3B-Instruct-v2 \
--train_datasets /mnt/hdd0/Kaggle/arc24/data/arc-agi_training_challenges.json output-from-examples-v1 \
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

## Results

## Conclusion

## Next steps

## TODO

- [ ] Experiment to validate that I can extend the context window of the model. At the beginning is a simple instruction, then a lot of distraction text. If the model has enough context length the task is trivial, otherwise is impossible.
- [ ] How can I add a chat template to a model?
- [ ] Can I reach the same validation results as Qwen?
- [ ] Datasets for long context fine-tuning. https://huggingface.co/blog/wenbopan/long-context-fine-tuning#long-text-data
