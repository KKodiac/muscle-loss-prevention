# muscle-loss-prevention

## ✨ Smart Trainer
<p align='center'>
<img width='300px' height='100px' src='./data/Logo.PNG'>
</p>

------------------------------------------  

## 설명
  
* #### data 폴더
  - 팀원들이 직접 찍은 운동 영상이 있어 mp_train을 제외하고 푸쉬하겠습니다.

* #### exercise_count(Model_training).py 파일
  - 운동의 횟수를 카운팅 해주는 모델입니다.  
  - 저장한 모델의 해당 파일 이름은 model.h5입니다.  
  
* #### create data(OpticalFlow_Farneback).py 파일  
  - 광학흐름 파넬백 변환 과정이 담겨있습니다.  
  - 변환된 데이터를 만들기 위해 기존의 운동 영상(input)으로 파넬백 변환 영상(output)을 획득합니다.  
  
* #### final simulation.py 파일  
  - model.h5, pushup.pkl, squat.pkl 파일이 같은 디렉토리에 있으면 실행됩니다.  
  - 웹캠에서 시뮬레이션을 할 수 있도록 만들어 놓았습니다.
  
* #### 최종 모든 과정.ipynb 파일
  - 위 과정들을 쥬피터 노트북으로 구성해 놓은 파일입니다.  
  - 추가적으로 Logistic 회귀 모델 학습 과정 또한 담겨 있습니다.  
  - 시뮬레이션 또한 실행할 수 있습니다.

------------------------------------------

|Push Up|Squat|
|:-:|:-:|
|![Push Up](./data/pushup.png?h=750&w=1260)|![Squat](./data/squat.jpg?h=750&w=1260)|

------------------------------------------  

## TODO
```
 - 파라미터 조정
 - 언더샘플링
 - 데이터 추가 및 학습
```