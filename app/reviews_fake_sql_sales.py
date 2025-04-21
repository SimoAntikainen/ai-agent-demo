from datetime import datetime, timedelta
from pathlib import Path
import random
import sqlite3
import csv
import ast

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "book_reviews"
CSV_FILE = DATA_DIR / "balanced_books.csv"
db_name = THIS_DIR / ".chat_app_backend.sqlite"

# Connect to SQLite
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Drop the tables if they exist
cursor.execute("DROP TABLE IF EXISTS Books")
cursor.execute("DROP TABLE IF EXISTS Products")
cursor.execute("DROP TABLE IF EXISTS Sales")

# Create the Books table (NOW includes productId)
cursor.execute("""
CREATE TABLE Books (
    productId TEXT,
    title TEXT,
    description TEXT,
    authors TEXT,
    image TEXT,
    preview_link TEXT,
    publisher TEXT,
    published_date TEXT,
    info_link TEXT,
    categories TEXT,
    ratings_count REAL
)
""")

# Create Products table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Products (
    product_id TEXT PRIMARY KEY,
    name TEXT,
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
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT,
    sale_date DATE,
    store_location TEXT,
    units_sold INTEGER,
    unit_price REAL,
    discount_applied REAL,
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
)
''')

# Load CSV and insert rows into Books
with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        try:
            authors = ", ".join(ast.literal_eval(row["authors"])) if row["authors"] else ""
        except Exception:
            authors = row["authors"]

        try:
            categories = ", ".join(ast.literal_eval(row["categories"])) if row["categories"] else ""
        except Exception:
            categories = row["categories"]

        cursor.execute("""
            INSERT INTO Books (
                productId, title, description, authors, image, preview_link,
                publisher, published_date, info_link, categories, ratings_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["productId"],
            row["Title"],
            row["description"],
            authors,
            row["image"],
            row["previewLink"],
            row["publisher"],
            row["publishedDate"],
            row["infoLink"],
            categories,
            float(row["ratingsCount"]) if row["ratingsCount"] else None
        ))

conn.commit()
print("✅ Data from balanced_books.csv successfully loaded into SQLite database.")

# Load books for product + sales creation
cursor.execute("SELECT productId, title, publisher, published_date FROM Books")
books = cursor.fetchall()

stores = ["New York", "San Francisco", "London", "Berlin", "Tokyo"]

for product_id, title, publisher, published_date in books:
    price = round(random.uniform(8.99, 29.99), 2)
    stock_quantity = random.randint(50, 500)
    is_active = True

    # Insert into Products using productId
    cursor.execute('''
        INSERT INTO Products (product_id, name, brand, release_date, price, stock_quantity, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (product_id, title, publisher, published_date, price, stock_quantity, is_active))

    # Generate monthly sales
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 4, 1)
    current_date = start_date

    while current_date <= end_date:
        units_sold = max(0, int(random.gauss(100, 40)))
        store_location = random.choice(stores)
        discount = round(random.uniform(0, 0.3), 2)

        cursor.execute('''
            INSERT INTO Sales (product_id, sale_date, store_location, units_sold, unit_price, discount_applied)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            product_id,
            current_date.strftime("%Y-%m-%d"),
            store_location,
            units_sold,
            price,
            discount
        ))

        current_date += timedelta(days=30)

conn.commit()

# Show top sellers
print("\n📈 Top 5 Best-Selling Books by Total Units Sold:\n")
cursor.execute("""
    SELECT P.name, SUM(S.units_sold) AS total_sold
    FROM Sales S
    JOIN Products P ON P.product_id = S.product_id
    GROUP BY S.product_id
    ORDER BY total_sold DESC
    LIMIT 5
""")

for row in cursor.fetchall():
    print(f"📚 {row[0]} — {row[1]} units sold")

conn.close()