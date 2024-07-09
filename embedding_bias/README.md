## 실험 결과
1. 기본 TF-IDF
   - 불용어 제거하지 않음
   - 명사만 활용
     ![heatmap_simple_tfidf0](https://github.com/4rldur0/AIFFEL-Quest/assets/111371565/f13eb847-c31b-463f-a5f8-9fe6fec311c1)
   - 기타, 다큐멘터리, 멜로로맨스, 뮤지컬, 성인물 장르가 편향성이 크다
       
2. TF-IDF + 불용어 제거
   ![heatmap_simple_tfidf](https://github.com/4rldur0/AIFFEL-Quest/assets/111371565/93534549-e9f5-409d-9e72-f7955e654ac4)
   - 1과 비교해 기타와 다큐멘터리에 대한 편향이 감소했다. 두 장르 다 일반적인 언어를 많이 사용하다보니 불용어에 더 민감하게 반응한 듯 하다

4. TF-IDF + 불용어 제거 + 동사
  ![heatmap_simple_tfidf_verb](https://github.com/4rldur0/AIFFEL-Quest/assets/111371565/a66d948f-a1d5-419a-9c24-3d1e52093219)
   - 특히 멜로로맨스에 대한 편향이 다른 실험에 비해 크게 감소했음을 알 수 있다

5. DTM + 불용어 제거
   ![heatmap_count](https://github.com/4rldur0/AIFFEL-Quest/assets/111371565/b7cec54a-d3f3-4d33-9c10-230c85706d5a)
   - 2와 비교했을 때 미미한 수치 변화를 제외하고는 비슷한 양상을 보인다. 따라서 이 데이터셋에 대해서는 TF-IDF와 DTM 간의 차이가 거의 없음을 알 수 있다  

6. LDA
   - 각각의 document에 같은 topic이 할당되는 문제가 계속 발생
   - LDA 모델 초기화 시 `topic_word_prior`, `doc_topic_prior`와 같은 하이퍼파라미터 조정했지만 해결되지 않음
   - target에서 두 문서의 중복 단어를 완전히 배제하여 LDA를 적용한 후 각각에 중복 단어를 일부 추가해보는 방법을 사용하려 했지만, 중복 단어를 없애더라도 같은 topic이 할당

## 코더 회고
- 배운 점: 벡터화 적용과 WEAT 점수 계산 방법에 대해 배울 수 있었다
- 아쉬운 점: LDA가 효과적이지 않아 아쉽다
- 느낀 점: 임베딩 편향은 앞으로 점점 더 중요해질 연구 주제라고 생각된다. WEAT 말고도 다른 편향 측정방법에 대해 더 공부해보고 싶다
- 어려웠던 점: 실험 결과를 볼 수는 있는데 각각의 결과가 왜 나왔는 지 해석하기가 어려웠다

---
## 리뷰어 
🔑 **PRT(Peer Review Template)**

- [o]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - [o]  문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - [o]  문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/9c81d3b8-5347-41ee-828f-faaacf262ecf)
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/9285b41a-26dd-43c4-8d1c-18df5722f1ee)
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/123d7c23-1b32-4390-a53e-989122231829)
    - [o]  해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/1af7184e-828f-41e9-aab2-9f936ad197f7)

해당일자 프로젝트에서 요구한 모든 루브릭을 완료하여 코드와 기록으로 남겨져 있었습니다.

- [o]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [o]  모델 선정 이유
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/e82f5c9e-93d5-443d-af22-4d2ed7232ddc)
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/358573ce-4a98-402f-a672-8c9a7eae826b)
    - []  Metrics 선정 이유
    - []  Loss 선정 이유
오늘 루브릭에서는 매트릭스와 loss가 필요하지 않았지만 stopword와 LDA등 다양한 방식으로 문제 접근을 수행했습니다.

- [o]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [o]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
    - [o]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/f4b67bfd-d93b-487c-b291-a5c56361c74a)
    - [o]  각 실험을 시각화하여 비교하였나요?
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/ee41be9e-71f6-4c1d-9f03-a9bbb58225a3)
    - [o]  모든 실험 결과가 기록되었나요?
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/83861c44-e8c2-4522-b7ea-42e5582fa456)
여러가지의 실험을 진행했으며 내동들을 상세히 기록했고 시각화를 통해 다양한 실험들의 결과를 저장했습니다.

- [o]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [o]  배운 점
    - [o]  아쉬운 점
    - [o]  느낀 점
    - [o]  어려웠던 점
    - ![image](https://github.com/4rldur0/AIFFEL-Quest/assets/132184507/46fc4b52-ca0e-44e7-b846-2d009e7dce90)

