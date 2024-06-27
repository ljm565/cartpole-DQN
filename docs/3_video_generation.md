# Video Generation
Here, we provide a guide for generating video from the trained cart pole model.

### 1. Video Generation
#### 1.1 Arguments
There are several arguments for running `src/run/video_generate.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to generate video. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to visualize.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `last`: Resume the model saved at the last epoch.


#### 1.2 Command
`src/run/video_generate.py` file is used to generate cart pole video of the model with the following command:
```bash
python3 src/run/video_generate.py --resume_model_dir ${project}/${name}
```