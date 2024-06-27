# Training DQN Cart Pole Model
Here, we provide guides for training a DQN cart pole model.

### 1. Configuration Preparation
To train a DQN cart pole model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu                 # examples: [0], cpu, mps... 

# project config
project: outputs/DQN
name: cartpole

# model config
hidden_dim: 256

# train config
batch_size: 64
episodes: 600
lr: 1e-3

# agent action config
eps_start: 0.9
eps_end: 0.05
eps_decay: 200
gamma: 0.8
target_update_duration: 10

# logging config
common: ['loss', 'duration', 'lr']
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `last`: Resume the model saved at the last epoch.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```
When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.