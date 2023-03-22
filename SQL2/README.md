# Intermediate commands in SQL


## Change decimal points
change minutes to 4 decimal places and mb to  2 decimal places
```
SELECT 
	name, 
  ROUND(milliseconds/60000.0, 2) AS minutes , 
  ROUND(bytes/ (1024*1024.0), 4) AS mb
from tracks ;
```
![image](https://user-images.githubusercontent.com/85028821/212546747-d9594af0-4d2d-4751-845a-df8b02c20041.png)

## Filter
filter with WHERE clause and IN
```
SELECT firstname, lastname, country 
from customers
WHERE country IN ('USA', 'Canada', 'United Kingdom') ;
```
![image](https://user-images.githubusercontent.com/85028821/212547013-252eb563-9f92-4866-8681-820f51acb37c.png)

filter with LIKE and wild card
```
SELECT firstname, lastname, email, country 
from customers
where email like '%@gmail.com' and country <> 'USA' ; --<> is not include
```
![image](https://user-images.githubusercontent.com/85028821/212548518-4fd0b748-1703-49d3-ac50-0fa0f74e1ba0.png)

## Join table
Join table with where clause and filter
```
SELECT 
    A.artistid,
    A.name As artistName,
    B.title AS albumName
FROM artists A, albums B
where A.artistid = B.artistid 
    and A.artistid in (5,20,100,115) ;
```

Join table with join clause and filter
```
SELECT 
    A.artistid,
    A.name As artistName,
    B.title AS albumName
FROM artists A 
join albums B on A.artistid = B.artistid 
    and A.artistid in (5,20,100,115) ;
```
![image](https://user-images.githubusercontent.com/85028821/212548855-88cc8a7f-fba8-4f0a-8b9c-570ee8bb4282.png)

Join table with JOIN ON vs JOIN USING
```
SELECT
    A.artistid,
    A.name AS artistName,
    B.title AS albumName,
    C.name AS trackName,
    D.name AS genrename
from artists A
join albums B on A.artistid = B.artistid
join tracks C ON B.albumid = C.albumid
join genres D ON C.genreid = D.genreid ;
```
```
SELECT
    A.artistid,
    A.name AS artistName,
    B.title AS albumName,
    C.name AS trackName,
    D.name AS genrename
from artists A
join albums B USING(artistid)
join tracks C USING(albumid)
join genres D USING(genreid) ;
```
![image](https://user-images.githubusercontent.com/85028821/212548966-1474591e-3f59-4ff1-968d-af6d6c10ef07.png)

## Subqueries
subqueries and select unique artistname that 1st character is L
```
SELECT DISTINCT artistName from (
  SELECT
      A.artistid,
      A.name AS artistName,
      B.title AS albumName,
      C.name AS trackName,
      D.name AS genrename
  from artists A
  join albums B USING(artistid)
  join tracks C using(albumid)
  join genres D using(genreid)
  WHERE A.name LIKE 'L%' 
) ;
```
select only customers in USA and Canada
```
-- subquery โดยจะ run inner query ก่อน outer query
SELECT firstname, lastname, email FROM (
	SELECT * FROM customers where country IN ('USA', 'Canada')
  );
```
![image](https://user-images.githubusercontent.com/85028821/226847902-49346ca6-f132-4a9f-9176-d212ee073b67.png)

subqueries with pipes
```
SELECT firstname ||' '|| lastname || 'uses email:' || email || 'for contact' || '.' AS statement
FROM (
	SELECT * FROM customers where country IN ('USA', 'Canada')
  );
```
![image](https://user-images.githubusercontent.com/85028821/226848363-1a0cb9a7-c942-4561-a21b-65ec2212e6b2.png)

The benefit of subqueries is that it runs faster because it queries data from the inner before.
```
-- subqueries (ข้อดีของการ filter table ข้างใน inner query ก่อนคือจะทำให้ดึงข้อมูลที่จะมา join น้อยลง จะทำให้ run ได้เร็วขึ้น
SELECT country, SUM(total) FROM customers 
JOIN (
      SELECT 
      customerid, 
      invoicedate, 
      total 
      FROM invoices
      WHERE STRFTIME('%Y', invoicedate) = '2010'
  ) AS sub
ON customers.customerid = sub.customerid
WHERE country IN ('USA', 'Canada', 'United Kingdom')
GROUP BY country;
```
![image](https://user-images.githubusercontent.com/85028821/226849768-84f783eb-462b-4fdf-9c46-d4b6455ee6c5.png)

## Filter NULL
filter all row that is not NULL in company column
```
SELECT * from customers
where company IS NOT NULL ; -- or use IS NULL
```

use COALESCE to replace NULL with desired word
```
SELECT 
  company, 
  COALESCE(company, 'End Customers') AS clean_company
from customers
```
![image](https://user-images.githubusercontent.com/85028821/212551183-07738594-a14a-4a51-8cb6-4ae927c87ba3.png)

## If else using CASE WHEN
COALESCE and CASE WHEN
```
SELECT 
  company, 
  COALESCE(company, 'End Customers') AS clean_company,
  CASE 
    	WHEN company is NULL then 'End Customers'
    	ELSE  'Corporate'
  END AS 'CASE WHEN'
from customers; 
```
![image](https://user-images.githubusercontent.com/85028821/212552100-f34e2226-9a75-4d74-8fd9-b22f99bcbc21.png)

CASE WHEN > 1 conditions
```
SELECT 
  firstname,
  email,
    CASE
      WHEN email like '%@gmail%' THEN 'Google Account'
      WHEN email like '%@yahoo%' THEN 'Yahoo Account'
      ELSE 'Other Email Providers'
    END 'Email_Group'
FROM customers;
```
![image](https://user-images.githubusercontent.com/85028821/212552191-5a2d0242-038e-44ea-9032-4e2c4109bbd8.png)

## Aggregate functions ignore NULL
```
-- aggregate functions ignore NULL
SELECT 
    AVG(bytes) avg_bytes, 
    SUM(bytes), 
    MIN(bytes), 
    MAX(bytes), 
    COUNT(bytes)
from tracks
```
![image](https://user-images.githubusercontent.com/85028821/226843589-acc2372b-8259-45a3-bbd3-89304171e67f.png)

## Group by With 2 columns
```
SELECT
  country,
  state,
  COUNT(*)
from customers
group by country, state
```
![image](https://user-images.githubusercontent.com/85028821/226845103-78801ad8-41c2-4517-b219-fd249ff0faec.png)

## Having
It uses for filter data after group by
```
SELECT 
  genres.name,
  COUNT(*) n_songs
from genres
join tracks on genres.genreid = tracks.genreid
group by genres.name
having n_songs >= 300 ;
```
![image](https://user-images.githubusercontent.com/85028821/226845714-c5c96111-cfd0-4665-8d7b-93ee086773ac.png)

## Having vs where
To exclude Rack genre after group by using Having
```
-- use having
SELECT 
  genres.name,
  COUNT(*) n_songs
from genres
join tracks on genres.genreid = tracks.genreid
group by genres.name
having genres.name <> 'Rock' -- not include Rock genre
Order by n_songs DESC  --descending order
Limit 5 ;
```
To exclude Rack genre before group by using Where
```
-- use where
SELECT 
    genres.name,
    COUNT(*) n_songs
from genres
join tracks on genres.genreid = tracks.genreid
where genres.name <> 'Rock' -- not include Rock genre
group by genres.name
Order by n_songs DESC  --descending order
Limit 5 ;
```
![image](https://user-images.githubusercontent.com/85028821/226846803-e442de25-8c83-4b39-adbe-de54420a5160.png)

use both where and having
```
SELECT 
    genres.name,
    COUNT(*) n_songs
FROM genres
JOIN tracks on genres.genreid = tracks.genreid
WHERE genres.name <> 'Rock' -- not include Rock genre
GROUP BY genres.name
HAVING n_songs < 400
ORDER BY n_songs DESC  --descending order
LIMIT 5 ;
```

## Aggregation function with Group by
```
SELECT
  genres.name,
  COUNT(*) n_songs,
  SUM(tracks.bytes) total_bytes,
  AVG(tracks.bytes) avg_bytes
FROM genres 
JOIN tracks ON genres.GenreId = tracks.GenreId
WHERE genres.name <> 'Rock' -- not include Rock genre
GROUP BY genres.name
HAVING n_songs < 400 -- filter groups after group by
ORDER BY n_songs DESC 
LIMIT 5; -- descending order
```
![image](https://user-images.githubusercontent.com/85028821/226847374-91549cee-62ba-4549-a844-a66e5fe83862.png)

## NTILE
Create a segment from column
```
-- create segment from bytes
SELECT 
     trackid, 
     name, 
     bytes,
     NTILE(10) OVER(ORDER BY bytes) AS bytes_segment
FROM tracks
```
![image](https://user-images.githubusercontent.com/85028821/226850658-9cf9ddd6-4e58-4b6f-bb5a-4f0e7f3fb075.png)

## View
Create temporary table, Do not interfere with the Database
```
CREATE VIEW my_veiw
AS
select firstname, lastname, company
from customers

SELECT * FROM my_view
```
![image](https://user-images.githubusercontent.com/85028821/226851587-6915503f-e4d1-4279-9e91-4a0ab9d3b726.png)

## Upper & Substr
Upper for change string to upper test, Substr for bring string (column, start index, no.)
```
SELECT 
    firstname, 
    lastname,
    firstname || ' ' || lastname  AS fullName,
    UPPER(firstname) || SUBSTR(lastname, 1, 1) ||  '@fullstack.com' AS email 
FROM customers;
```
![image](https://user-images.githubusercontent.com/85028821/226856099-06a8d067-08a4-4248-bbfe-21543a0c9a2f.png)

## Round
for round decimal numbers or can set the number of decimal points
```
SELECT 
    name, 
    ROUND(milliseconds/ 60000.0, 2)    AS minutes, -- value function 
    ROUND(bytes/ (1024*1024.0) , 4)    AS mb
FROM tracks;
```
![image](https://user-images.githubusercontent.com/85028821/226857316-b6ae2d61-a5ff-42d7-aaae-cd706ad94d7e.png)

## CAST
cast used to convert data type in SQL, Change only query Not change in Database
```
SELECT 
  CAST('100' AS INT), 
  TYPEOF(CAST('100' AS INT)), 
  CAST(100 AS TEXT),
  TYPEOF(CAST('100' AS TEXT)) 
```
![image](https://user-images.githubusercontent.com/85028821/226859654-0a7325a3-1bdf-4fd4-a3d1-d20a22d89a7b.png)

## STRFTIME
work with date-time
```
-- date format YYYY-MM-DD 
SELECT 
    invoicedate,
    STRFTIME('%Y', invoicedate) AS YEAR,
    STRFTIME('%m', invoicedate) AS month,
    STRFTIME('%d', invoicedate) AS day,
    STRFTIME('%Y-%m', invoicedate) AS monthID
from invoices 
WHERE monthID = '2013-06' ;
```
![image](https://user-images.githubusercontent.com/85028821/226858496-d25e7be0-2627-4076-b86a-69cb20b72518.png)

use STRFTIME with CAST
```
-- format date column (actually it's text in SQLite)
SELECT 
  invoicedate, 
  CAST(STRFTIME('%Y', invoicedate) AS INT)  AS year,
  CAST(STRFTIME('%m', invoicedate) AS INT)  AS month,
  CAST(STRFTIME('%d', invoicedate) AS INT)  AS day,
  STRFTIME('%Y-%m', invoicedate)   AS monthID
FROM invoices WHERE year = 2013 AND month = 9;
```
![image](https://user-images.githubusercontent.com/85028821/226860507-5751de79-47e4-4657-9210-63f592da6361.png)
