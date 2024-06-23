# Cart Pole DQN
## 설명
본 코드는 [Gymnasimum](https://gymnasium.farama.org/)의 gym 라이브러리를 이용하여 Deep [Q Network (DQN)](https://arxiv.org/pdf/1312.5602.pdf)을 바탕으로 Cart Pole 강화학습을 수행합니다. DQN은 기존 Q-table을 바탕으로 Q learning을 하던 고전적인 방법을 Q-table을 딥러닝 모델로 대체한 딥러닝 기반의 강화학습 알고리즘입니다.
Cart Pole DQN에 대한 설명은 [DQN을 이용한 Cart Pole 세우기](https://ljm565.github.io/contents/dqn2.html)을 참고하시기 바랍니다.
<br><br><br>

## 모델 종류
* ### DQN
    DQN을 바탕으로 Cart Pole 강화학습을 수행합니다.
<br><br>


## 사용 데이터
* gym==0.23.1
* Cart Pole v1
<br><br><br>


## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, test}, **필수**: 학습을 시작하려면 train, 학습된 모델을 가지고 있어서 cart pole의 강화학습 gif 결과를 보고싶은 경우에는 test로 설정해야합니다. test 모드를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m {test} 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, test를 할 경우에도 실험할 모델의 이름을 입력해주어야 합니다(최초 학습시 src/config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 src/main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 src/main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 최종 학습 된 모델의 test set에 대한 결과를 확인할 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/src/config.json이 아닌, base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 src/main.py -d cpu -m test -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/src/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * base_path: 학습 관련 파일이 저장될 위치.
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/src/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/src/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * hidden_dim: DQN 모델의 hidden dimension.
    * batch_size: batch size 지정.
    * episodes: 학습 episode 설정.
    * lr: learning rate 지정.
    * eps_start, eps_end, eps_decay: Agent의 action 선택 시 사용하는 threshold 계산을 위한 파라미터.
    * gamma: 미래 Q value에 대한 감쇠율.
    * target_update_duration: Target network를 학습되는 Q network로 복사하는 episode 주기.
    <br><br><br>


## 결과
* ### Cart Pole Duration History
    <img src="docs/figs/duration.png" width="80%"><br><br>
    gym 라이브러리에 의해 최대 500 duration이 넘어가면 종료됩니다.

* ### Cart Pole 결과
    <img src="docs/figs/dqn_cartpole.gif" width="80%"><br><br>


<br><br><br>
