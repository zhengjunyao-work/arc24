# Iteration 50. Last trainings

_06-11-2024_

## Goal

Can we improve the LB score by training the last models on a powerful server?

## Motivation

There is not enough time to train models on Veridas cluster, I need to use a 8xA100 server to train
the models fast enough for the end of the challenge.

## Development

### Steps to train the model

1. Add the public SSH key of the machine to [Github](https://github.com/settings/keys). `cat ~/.ssh/id_rsa.pub`
1. Clone the arc 24 repo to the machine: `cd ~/code; git clone git@github.com:ironbar/arc24.git`
1. Create a python environment for training

```bash
cd ~/code/arc24
python3 -m virtualenv ~/envs/arc24
source ~/envs/arc24/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

1. Do some first trainings to see if the training speed is enough

```bash
source ~/envs/arc24/bin/activate
export gpus=8
export batch_size=16
export steps=200
export per_device_train_batch_size=1
export model_path=Qwen/Qwen2.5-0.5B-Instruct
export WANDB_API_KEY=

accelerate launch --num_processes ${gpus} --num_machines 1 --mixed_precision bf16 --multi_gpu \
/root/code/arc24/scripts/fine-tuning.py \
--n_gpus ${gpus}8 \
--batch_size ${batch_size} \
--per_device_train_batch_size ${per_device_train_batch_size} \
--output_dir /root/models/20241106_debug_training_speed/${gpus}XA100_bs${batch_size}_pdtbs${per_device_train_batch_size}_${steps}steps_$(basename $model_path)
--max_steps ${steps} \
--model_path ${model_path} \
--lora_r 128 \
--device_map None \
--verbose \
--max_seq_len 8192 \
--learning_rate 5e-5 \
--train_datasets /root/code/arc24/data/original_data/arc-agi_training_challenges.json output-from-examples-v1 \
--val_dataset /root/code/arc24/data/original_data/arc-agi_evaluation_challenges.json output-from-examples-v1 \
--remove_train_samples_to_fit_max_seq_len \
--save_steps 500 \
--eval_steps 5000000 \
--warmup_ratio 1e-1
```

## Results

## Conclusion

## Next steps

## TODO

- [ ]
