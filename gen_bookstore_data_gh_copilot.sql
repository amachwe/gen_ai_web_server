-- Insert Authors
INSERT INTO authors (author_id, name) VALUES
('A1', 'Author One'),
('A2', 'Author Two'),
('A3', 'Author Three');

-- Insert Books
-- Ensuring at least 2 books per author
INSERT INTO books (book_id, name, author_id) VALUES
('B1', 'Book One', 'A1'),
('B2', 'Book Two', 'A1'),
('B3', 'Book Three', 'A2'),
('B4', 'Book Four', 'A2'),
('B5', 'Book Five', 'A3'),
('B6', 'Book Six', 'A3'),
('B7', 'Book Seven', 'A3');

-- Insert Inventory
-- One record for each book
INSERT INTO inventory (inv_id, book_id, quantity) VALUES
('INV1', 'B1', 10),
('INV2', 'B2', 15),
('INV3', 'B3', 5),
('INV4', 'B4', 8),
('INV5', 'B5', 12),
('INV6', 'B6', 7),
('INV7', 'B7', 9);

-- Insert Transactions
-- Randomly associating them with books
INSERT INTO transactions (tx_id, book_id, quantity_sold, sale_date) VALUES
('TX1', 'B1', 1, '2023-04-01'),
('TX2', 'B2', 2, '2023-04-02'),
('TX3', 'B4', 1, '2023-04-03'),
('TX4', 'B5', 3, '2023-04-04'),
('TX5', 'B7', 2, '2023-04-05');