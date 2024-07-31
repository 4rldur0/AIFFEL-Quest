```
$docker compose up -d --build
```

![image](https://github.com/user-attachments/assets/a196777b-19d2-4727-83e7-c12d57fd008d)


#### 궁금한 점
> - `main.py`에서 했던 것처럼 fastapi와 streamlit을 각각 쓰레드로 만들어서 실행
> -  `main.py` 없이 `docker-compose.yaml`에 두 개의 service를 정의하여 `backend` service에서 fastapi를 실행하고, `frontend` service에서 streamlit을 실행

두  방법 중에 어떤 게 좋을까?
