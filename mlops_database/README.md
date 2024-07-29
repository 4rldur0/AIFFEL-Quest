### postgreSQL 데이터베이스에 새로운 테이블을 만들어서 scikit-learn의 iris 데이터를 한 행 씩 추가하는 예제

```
$docker compose up -d

# 로컬에서 데이터 확인
$PGPASSWORD=mypassword psql -h localhost -p 5432 -U myuser -d mydatabase
mydatabase=# select * from iris_data;

# 컨테이너 안에서 데이터 확인
$docker exec -it data-generator /bin/bash
#PGPASSWORD=mypassword psql -h postgres-server -p 5432 -U myuser -d mydatabase
mydatabase=# select * from iris_data;

$docker compose down -v
```
