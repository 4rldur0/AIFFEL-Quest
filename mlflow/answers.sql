#2
SELECT type, COUNT(*) AS cnt
FROM netflix_titles
GROUP BY type;
#3
select title
from netflix_titles
where director = "Luis Ara, Ignacio Jaunsolo";
#4 
SELECT title
FROM netflix_titles
WHERE date_added LIKE '2019-11-%';
INTO OUTFILE '/var/lib/mysql-files/201911.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
#5
SELECT title
FROM netflix_titles
WHERE title LIKE '%winter%';
