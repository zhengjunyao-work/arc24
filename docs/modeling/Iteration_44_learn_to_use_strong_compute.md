# Iteration 44. Learn to use Strong Compute

_22-10-2024_

## Goal

Learn to use the Strong Compute cluster.

## Motivation

[Strong compute](https://strongcompute.com/) has graciously granted me with compute credits to speedup my development in the
last weeks of the challenge. That means I will have access to GPUs with 80GBs of memory and I could
train much faster than with the [Veridas](https://veridas.com/en/) cluster.

## Development

- [Strong compute documentation](https://strong-compute.gitbook.io/developer-docs)
- [ISC demos repo](https://github.com/StrongResearch/isc-demos)
- [Strong compute Discord](https://discord.com/channels/1093337804051849296/1283175776165822556)

### Quick guide to connect to Strong compute

1. Go to [Strong compute control panel](https://cp.strongcompute.ai/user/credentials) and start a workstation.
1. Start the vpn with `sudo wg-quick up wg0`
1. Connect to the workstation using vscode
1. After all the work has been done disconnect the vpn `sudo wg-quick down wg0`
1. And stop the workstation

### Creating a python environment for the experiments

I already have the requirements on `requirements.txt` file, so I just have to clone the repo into
the workstation.

To do so I have created an ssh key doing `ssh-keygen` on the workstation and added the public key to github.

```bash
cd ~/code/arc24
python3 -m virtualenv ~/envs/arc24
source ~/envs/arc24/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### First training

```job.isc
isc_project_id = "46f4672b-2489-457f-b302-eab855b36b70"
experiment_name = "first_arc24_training"
gpu_type = "24GB VRAM GPU"
gpus = 8
compute_mode = "burst"
output_path = "~/outputs/first_arc24_training"
command = "source ~/envs/arc24/bin/activate && cd ~/code/arc24/scripts && ~/jobs/job.sh"
burst_shape_priority_list = ["oblivus-mon1-h100n"]
```

Better configuration with the whole command:

```toml
isc_project_id = "46f4672b-2489-457f-b302-eab855b36b70"
experiment_name = "first_arc24_training_a100n"
gpu_type = "24GB VRAM GPU"
gpus = 8
compute_mode = "burst"
dataset_id = "0cfd54a3-4096-494e-93d5-a073126e81e2"
output_path = "~/models/20241022_no_training/04_A100n_lora064-Qwen2.5-0.5B-Instruct_lr1e-4_bs16_10000steps_2gpus_8192msl"
burst_shape_priority_list = ["oblivus-mon1-a100n"]
command = '''
source ~/envs/arc24/bin/activate && 
source ~/jobs/secrets.sh && 
accelerate launch --num_processes 8 --num_machines 1 --mixed_precision bf16 --multi_gpu 
~/code/arc24/scripts/fine-tuning.py 
--max_steps 10000 
--model_path=Qwen/Qwen2.5-0.5B-Instruct 
--lora_r 64 
--output_dir $(dirname "${OUTPUT_PATH}")
--n_gpus=8 
--batch_size=16 
--device_map None 
--no-verbose 
--compose_new_task_probability 0.5 
--compose_new_task_weights 1 1 1 1 
--max_seq_len 8192 
--learning_rate=1e-4 
--train_datasets ~/code/arc24/data/original_data/arc-agi_evaluation_challenges.json output-from-examples-v1 
--train_datasets ~/code/arc24/data/external_data/kaggle.json output-from-examples-v1  
--train_datasets ~/code/arc24/data/external_data/pqa-dataset-1k.json output-from-examples-v1  
--train_datasets ~/code/arc24/data/external_data/neoeye_tama.json output-from-examples-v1  
--train_datasets ~/code/arc24/data/external_data/MINI-ARC.json output-from-examples-v1  
--train_datasets ~/code/arc24/data/original_data/arc-agi_evaluation_challenges.json input-from-inputs-v0 
--train_datasets ~/code/arc24/data/external_data/kaggle.json input-from-inputs-v0  
--train_datasets ~/code/arc24/data/external_data/pqa-dataset-1k.json input-from-inputs-v0  
--train_datasets ~/code/arc24/data/external_data/neoeye_tama.json input-from-inputs-v0  
--train_datasets ~/code/arc24/data/external_data/MINI-ARC.json input-from-inputs-v0  
--val_dataset ~/code/arc24/data/original_data/arc-agi_training_challenges.json output-from-examples-v1 
--remove_train_samples_to_fit_max_seq_len 
--eval_steps=200 
--warmup_ratio 1e-1'''
```

### Initial idea

- Since my datasets are small I believe I can work on the root folder.
- They have said that it only has sense to use GPUs in multiples of 8.
- H100 is [newer and faster than a100](https://gcore.com/blog/nvidia-h100-a100/)

### H100 vs A100

https://oblivus.com/pricing/

![pricing differences](res/2024-10-23-11-38-07.png)

The Nvlink machines are slightly more expensive than the pcie. For multi-gpu training it should be
faster so probably it's better to just avoid using the PCIE machines.

The H100 is more expensive than the A100, we have to see if the speedup is worth it.

### Debugging burst errors

We can find a `.tar.zst` file in the exports folder. We should copy it first to a different folder
because the exports folder is a fused folder. Then we can untar it.

```
cp exports/183e895a-bbb1-4e3a-b9e8-f3ee02c5e5cb.tar.zst copied_exports
apt-get install -y zstd
tar --use-compress-program=unzstd -xvf 183e895a-bbb1-4e3a-b9e8-f3ee02c5e5cb.tar.zst
```

## Results

First training with 8xA100 is 5x times faster than using 2xA6000.

However in the following days I have been unable to run new trainings, and finally after 2 days
of struggling I have managed to run two jobs (on A100 and H100) but they were 4 times slower than the
previous run and they have finished with `strong_error` after less than 2 hours of running.

Today is Sunday and I have been able to run 3 fast trainings, one slow. I don't see any speed difference
between A100 and H100. I have tried using a batch size of 2 per GPU but it is not faster.

## Conclusion

So far it seems that Strong Compute cluster is very unstable. But we have seen that we can train at
least 5 times faster in a machine with 8xA100. So we could go directly to a cloud provider and do a fast
training if necessary.

## Next steps

## TODO

- [ ] Create a python environment for the experiments
- [ ] Copy the data and code to the ISC (instant super computer)
- [ ] Train a submission model with strong compute
- [ ] How much faster can I train?
- [ ] Differences between A100 and H100
- [ ] https://huggingface.co/docs/accelerate/en/usage_guides/low_precision_training On H100
- [ ] Multi-line submit files TOML https://toml.io/en/