🔑 **PRT(Peer Review Template)**
리뷰어: 김서우

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
        1. 모델과 데이터를 정상적으로 불러오고, 작동하는 것을 확인하였다.
        2. ![image](https://github.com/user-attachments/assets/9294b060-1498-4245-af24-293eec7d00df)

        3. Preprocessing을 개선하고, fine-tuning을 통해 모델의 성능을 개선시켰다.

        5. 모델 학습에 Bucketing을 성공적으로 적용하고, 그 결과를 비교분석하였다.
           ![image](https://github.com/user-attachments/assets/16d102e4-8ad6-4005-9efe-18069b757f8c)

        => 모델 데이터 불러오고, 전처리를 하고, bucketing 및 결과 분석까지 모두 잘 하셨습니다. 그리고 중간에 보이는 torch.cude.empty_cache() 로 out of memory 현상도 잘 해결해주셨습니다. 이 부분 궁금했는데 개인적으로 알아가서 좋았아요!! 

- [x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [x]  모델 선정 이유
          ![image](https://github.com/user-attachments/assets/27262be8-56ee-4ac9-930b-294bfd0a4e21)

    - [x]  Metrics 선정 이유
          ![image](https://github.com/user-attachments/assets/665fba80-c962-42da-9681-4c43c680f9ef)

    - [x]  Loss 선정 이유

    - 하면서 궁금했던 점에 대해서 주석을 달아놓거나
       ![image](https://github.com/user-attachments/assets/6d51a89b-3209-40aa-8c56-786e5ead07d9)
  

- [x]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [x]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
              ![image](https://github.com/user-attachments/assets/3f5fc2d9-fec3-46a7-8b96-fb02f8cbdca2)

    - [x]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
              ![image](https://github.com/user-attachments/assets/1fdbe3e0-cb05-457d-8a52-bcf0434b2246)

    - [x]  각 실험을 시각화하여 비교하였나요?+ 모든 실험 결과가 기록되었나요?
          ![image](https://github.com/user-attachments/assets/1a663416-4a43-464c-b48f-143d6cfcc788)


- [x]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [x]  배운 점
    - [x]  아쉬운 점
    - [x]  느낀 점
    - [x]  어려웠던 점
          ![image](https://github.com/user-attachments/assets/78457e24-557b-4a13-8ec5-c08508ccc217)

