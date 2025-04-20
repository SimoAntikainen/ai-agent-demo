from pathlib import Path
import sqlite3


THIS_DIR = Path(__file__).parent
db_name = THIS_DIR / '.chat_app_backend.sqlite'

# Connect to SQLite database (or create it)
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create Products table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    brand TEXT,
    release_date DATE,
    price REAL,
    stock_quantity INTEGER,
    is_active BOOLEAN
)
''')

# Create Sales table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Sales (
    sale_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    sale_date DATE,
    store_location TEXT,
    units_sold INTEGER,
    unit_price REAL,
    discount_applied REAL,
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
)
''')

# Insert mock data into Products
products_data = [
    (1, 'UltraPhone 15', 'Smartphone', 'TechPro', '2024-09-15', 999.99, 500, True),
    (2, 'SmartWatch X2', 'Smartwatch', 'WristIQ', '2024-06-01', 299.99, 150, True),
    (3, 'NoiseCanceller+', 'Headphones', 'SoundMax', '2023-11-20', 199.99, 120, False),
    (4, 'Laptop Z5', 'Laptop', 'ByteTech', '2025-02-10', 1299.99, 75, True),
    (5, 'ProBuds Air', 'Earbuds', 'SoundMax', '2024-03-25', 149.99, 200, True)
]
cursor.executemany('''
INSERT INTO Products (product_id, name, category, brand, release_date, price, stock_quantity, is_active)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', products_data)

# Insert mock data into Sales
sales_data = [
    (1001, 1, '2025-01-10', 'New York', 2, 999.99, 0.00),
    (1002, 2, '2025-02-14', 'Los Angeles', 1, 279.99, 20.00),
    (1003, 3, '2024-12-05', 'Chicago', 3, 179.99, 20.00),
    (1004, 1, '2025-03-01', 'Houston', 1, 949.99, 50.00),
    (1005, 5, '2025-04-10', 'Seattle', 4, 149.99, 0.00)
]
cursor.executemany('''
INSERT INTO Sales (sale_id, product_id, sale_date, store_location, units_sold, unit_price, discount_applied)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', sales_data)

# Commit and close connection
conn.commit()
conn.close()