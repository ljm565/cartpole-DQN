# Cart Pole Training using Deep Q Network

## Introduction
본 코드는 [Gymnasimum](https://gymnasium.farama.org/)의 gym 라이브러리를 이용하여 Deep [Q Network (DQN)](https://arxiv.org/pdf/1312.5602.pdf)을 바탕으로 Cart Pole 강화학습을 수행합니다. DQN은 기존 Q-table을 바탕으로 Q learning을 하던 고전적인 방법을 Q-table을 딥러닝 모델로 대체한 딥러닝 기반의 강화학습 알고리즘입니다.
Cart Pole DQN에 대한 설명은 [DQN을 이용한 Cart Pole 세우기](https://ljm565.github.io/contents/dqn2.html)을 참고하시기 바랍니다.
<br><br><br>

## Supported Models
### DQN
* DQN이 구현 되어있습니다.
<br><br><br>


## Base Dataset
* 실험으로 사용하는 데이터는 `gym==0.23.1`의 Cart Pole v1 입니다.
<br><br><br>


## Supported Devices
* CPU, GPU, MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>

## Project Tree
본 레포지토리는 아래와 같은 구조로 구성됩니다.
```
├── configs                          <- Config 파일들을 저장하는 폴더
│   └── *.yaml
│
└── src      
    ├── models
    |   └── dqn.py                   <- DQN 모델 파일
    |
    ├── run                   
    |   ├── train.py                  <- 학습 실행 파일
    |   └── video_generate.py         <- 비디오 생성 파일
    |
    ├── tools                   
    |   ├── model_manager.py          
    |   └── training_logger.py        <- Training logger class 파일
    |
    ├── trainer                 
    |   ├── build.py                  <- 모델 등을 정의하는 파일
    |   └── trainer.py                <- 학습, 비디오 생성 하는 클래스 파일
    |
    └── uitls                   
        ├── __init__.py               <- Logger, 버전 등을 초기화 하는 파일
        ├── filesys_utils.py       
        ├── model_utils.py       
        └── training_utils.py     
```
<br><br>


## Tutorials & Documentations
DQN cart pole 모델 학습을 위해서 다음 과정을 따라주시기 바랍니다.
1. [Getting Started](./1_getting_started_ko.md)
2. [Training](./2_trainig_ko.md)
3. ETC
   * [Video Generation](./3_video_generation_ko.md)

<br><br><br>


## Training Results
### Cart Pole Duration History
<img src="figs/duration.png" width="80%"><br><br>
gym 라이브러리에 의해 최대 500 duration이 넘어가면 종료됩니다.

### Cart Pole Training Results
<img src="figs/dqn_cartpole.gif" width="80%"><br><br>


<br><br>
