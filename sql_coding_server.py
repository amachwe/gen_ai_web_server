import llm_server as llm_server

if __name__ == "__main__":

    import transformers

    MODEL_ID = "defog/llama-3-sqlcoder-8b"

    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    ## Configuration for the server including the prompting hint
    config = {
        "prompting_hint":  """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:

CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
"""
    }
    ## Create the server
    server = llm_server.LLM_Server(llm_server.LLM_Server_Pipe_Wrapper(MODEL_ID, tokenizer, model,  config))   

    server.start()