# Video Generation
여기서는 학습된 DQN cart pole 모델의 결과 비디오를 생성하는 가이드를 제공합니다.

### 1. Video Generation
#### 1.1 Arguments
`src/run/video_generate.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-r`, `--resume_model_dir`]: Confusion matrix 가시화를 할 모델의 경로. `${project}/${name}`까지의 경로만 입력하면, 자동으로 `${project}/${name}/weights/`의 모델을 선택하여 모델을 로드합니다.
* [`-l`, `--load_model_type`]: [`metric`, `last`] 중 하나를 선택.
    * `metric`(default): Valdiation accuracy가 최대일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.


#### 1.2 Command
`src/run/video_generate.py` 파일로 다음과 같은 명령어를 통해 학습된 모델의 cart pole video를 생성합니다.
```bash
python3 src/run/video_generate.py --resume_model_dir ${project}/${name}
```