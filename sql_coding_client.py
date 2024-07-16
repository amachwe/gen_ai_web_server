import requests
import json
import sqlite3
import os


def extract_sql(text: str)->str:
    """
    Extract the SQL code from the response from the LLM
    """
    return text.split("```sql")[1]

def execute_ddl(text: str, cursor: sqlite3.Cursor)->None:
    """
    Execute the DDL statements given an open cursor
    """
    for ddl in text.split("\n"):
        print("Executing (DDL): ", ddl)
        cursor.execute(ddl)
    cursor.connection.commit()
    

def execute_statements(stmts: str, cursor: sqlite3.Cursor)-> None:
    """
    Execute the SQL insert statements returned by the LLM given an open cursor
    """
    for line in stmts.split(";"):
        if line == "":
            continue
        text = line+";"
        print("Executing: ", text)
        try:
          cursor.execute(text)
          cursor.connection.commit()
        except sqlite3.OperationalError as oe:
          print("Error: ", oe,"\nStatement: ", text)
          # human in the loop
          statement = input("Enter the corrected statement or press Enter to ignore: ")
          if statement != "":
            cursor.execute(statement)
        

def execute_query(query: str, cursor: sqlite3.Cursor)->None:
   """
   Execute the SQL query returned by the LLM given an open cursor and print the results
   """
   for line in query.split(";"):
        if line == "":
            continue
        text = line+";"
        print("Executing: ", text)
        
        try:
          res = cursor.execute(text)
          for row in res.fetchall():
              print(row)
        except sqlite3.OperationalError as oe:
          print("Error: ", oe,"\nStatement: ", text)
          # human in the loop
          statement = input("Enter the corrected statement or press Enter to ignore: ")
          if statement != "":
            cursor.execute(statement)

def create_query(question: str, ddl: str)->str:
   """
   Helper method to create the Question prompt for the LLM using a template
   """

   return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:
{ddl}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
"""

def request_llm(prompt: str, headers: dict, request_url: str)->None:
    """
    Helper method to send a request to the LLM
    """
    print("Sent Query")
    response = requests.post(request_url, data=json.dumps({"prompt":prompt}), headers=headers).json()
    query = extract_sql(response.get("response"))
    print(query)
    execute_query(query, cur)


if __name__ == "__main__":
    
    
    ## Configuring the connection and headers
    DB_NAME = "db/sql_coding.db"
    TARGET_URL = "http://localhost:5000"
    REQUEST_URL = f"{TARGET_URL}/request"
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }

    ## Remove the database if it exists
    if os.path.exists(DB_NAME):
      os.remove(DB_NAME)

    ## Create the database and return a cursor to it
    cur = sqlite3.connect(DB_NAME).cursor()


    ## DDL to create the tables
    ddl = """
    create table authors (author_id varchar(20) primary key not null, name text not null);
    create table books (book_id varchar(20) primary key not null, name text not null, author_id varchar(20) not null, FOREIGN KEY(author_id) REFERENCES authors(author_id));
    create table inventory (inv_id varchar(20) primary key not null, book_id varchar(20) not null, quantity int not null, FOREIGN KEY(book_id) REFERENCES books(book_id));
    create table transactions(tx_id varchar(20) primary key not null, book_id varchar(20) not null, quantity_sold int not null, sale_date date not null, FOREIGN KEY(book_id) REFERENCES books(book_id));
"""

    ## Execute the DDL - no LLMs yet
    execute_ddl(ddl, cur)

    ## Questions to ask the LLM
    question1 = "Tell me which author has written the most books?"
    question2 = "Tell me which books had the highest sales?"
    question3 = "Tell me the name of the author that has the most sales?"

    ## Create the query prompts for the LLM based on the questions and the DDL
    prompt_query_auth_books = create_query(question1, ddl)
    
    prompt_query_book_sales = create_query(question2, ddl)

    prompt_query_most_sales_author = create_query(question3, ddl)

    
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
    syn_data = extract_sql(response.get("response"))
    print(syn_data)

    ## Execute the returned SQL
    execute_statements(syn_data, cur)

    ## Now that the tables are populated with the synthetic data, we can ask the LLM the questions, execute the SQL returned and print the results.
    request_llm(prompt_query_auth_books, headers, REQUEST_URL)
    request_llm(prompt_query_book_sales, headers, REQUEST_URL)
    request_llm(prompt_query_most_sales_author, headers, REQUEST_URL)




