# Iteration 36. Solving evaluation tasks with code

_09-10-2024_

## Goal

Can we solve the evaluation tasks by predicting code that implements the tasks?

## Motivation

On [Iteration 34](Iteration_34_developing_omni_arc.md) I trained models on omni-arc tasks. It was unclear
if the approach of `output-from-examples` benefited from training the model to do multiple tasks.

However if I can predict python code that could be game-changer because I can verify the python
code with the train samples.

## Development

First models were trained with 100 training tasks, second model with close to 150. Coverage of the
training dataset is important because it's likely correlated with coverage of the evaluation and test dataset.

### First steps with inference

<details>
  <summary>Click to see bash commands</summary>

```bash
# baseline
python inference.py \
--model_path  /mnt/hdd0/Kaggle/arc24/models/20241006_omniarc_validation/02_omni-arc-400-code-from-examples-Qwen2.5-0.5B-Instruct_lr5e-5_14000steps_2gpus_8192msl/checkpoint-14000 \
--prompt_version code-from-examples-v0 \
--dataset_path /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json \
--predictions_per_task 8 \
--output_filepath /mnt/hdd0/Kaggle/arc24/debug/first_predictions/checkpoint-14000/inference_evaluation_x008.json \
--verbose

python inference.py \
--model_path  /mnt/hdd0/Kaggle/arc24/models/20241006_omniarc_validation/02_omni-arc-400-code-from-examples-Qwen2.5-0.5B-Instruct_lr5e-5_14000steps_2gpus_8192msl/checkpoint-14000 \
--prompt_version code-from-examples-v0 \
--dataset_path /mnt/hdd0/Kaggle/arc24/data/arc-agi_evaluation_challenges.json \
--predictions_per_task 32 \
--output_filepath /mnt/hdd0/Kaggle/arc24/debug/first_predictions/checkpoint-14000/inference_evaluation_x032.json
```

</details>

The model is generating valid python code, I have to improve the inference script to check that the
code is correct and create the output. Add timeouts for safety.

## Results

### Preliminar results

I solve 4% of the tasks from the evaluation dataset. All predictions seem to be correct because they
are validated against the train dataset. When using temperature 0 there does not seem to be any favorable scaling law.

## Conclusion

## Next steps

## TODO

- [ ] How to execute the code safely and with timeouts? Check AIMO competition. This should be added
  to the omni-arc repo, because all the dsl functions are there.
- [ ] How does the method scale with compute? Validation should allow to scale well.
