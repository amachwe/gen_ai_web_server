
--  "Tell me which author has written the most books?"

SELECT a.name, COUNT(b.book_id) AS number_of_books
FROM authors a
JOIN books b ON a.author_id = b.author_id
GROUP BY a.name
ORDER BY number_of_books DESC
LIMIT 1;


--  "Tell me which books had the highest sales?"

SELECT b.name, SUM(t.quantity_sold) AS total_sales
FROM transactions t
JOIN books b ON t.book_id = b.book_id
GROUP BY b.name
ORDER BY total_sales DESC
LIMIT 1;


--  "Tell me the name of the author that has the most sales?"

SELECT a.name, SUM(t.quantity_sold) AS total_sales
FROM transactions t
JOIN books b ON t.book_id = b.book_id
JOIN authors a ON b.author_id = a.author_id
GROUP BY a.name
ORDER BY total_sales DESC
LIMIT 1;