import requests
import json
import sqlite3
import os

 ## Configuring the connection and headers
TARGET_URL = "http://localhost:5000"
REQUEST_URL = f"{TARGET_URL}/request"
headers = {
    "Content-Type": "application/json",
    "Connection": "keep-alive",
}

## DDL to create the tables
ddl = """
    create table authors (author_id varchar(20) primary key not null, name text not null);
    create table books (book_id varchar(20) primary key not null, name text not null, author_id varchar(20) not null, FOREIGN KEY(author_id) REFERENCES authors(author_id));
    create table inventory (inv_id varchar(20) primary key not null, book_id varchar(20) not null, quantity int not null, FOREIGN KEY(book_id) REFERENCES books(book_id));
    create table transactions(tx_id varchar(20) primary key not null, book_id varchar(20) not null, quantity_sold int not null, sale_date date not null, FOREIGN KEY(book_id) REFERENCES books(book_id));
"""

## create the synthetic data generation prompt
prompt_create_data = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Given DDL:
    {ddl}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
Generate data to populate the tables Authors (3 records), Books (7 records with at least 2 books per author), Inventory (7 records), and Transactions (5 records).

```sql
"""

 ## Send the synthetic data generation prompt to the LLM and print the response
print("Sent Syn Data")
response = requests.post(REQUEST_URL, data=json.dumps({"prompt":prompt_create_data, "process_logits": True}), headers=headers).json()

logits = response["logits"]

