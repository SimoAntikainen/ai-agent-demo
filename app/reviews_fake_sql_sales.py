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

# Drop the table if it exists
cursor.execute("DROP TABLE IF EXISTS Books")
cursor.execute("DROP TABLE IF EXISTS Products")
cursor.execute("DROP TABLE IF EXISTS Sales")

# Create the table
cursor.execute("""
CREATE TABLE Books (
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

# Create Product table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER,
    sale_date DATE,
    store_location TEXT,
    units_sold INTEGER,
    unit_price REAL,
    discount_applied REAL,
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
)
''')

# Load CSV and insert rows
with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        # Parse authors and categories from stringified list to comma-separated strings
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
                title, description, authors, image, preview_link,
                publisher, published_date, info_link, categories, ratings_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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



# Load books from existing books table
cursor.execute("SELECT title, publisher, published_date, categories FROM books")
books = cursor.fetchall()

# Prepare list of store locations
stores = ["New York", "San Francisco", "London", "Berlin", "Tokyo"]

# Generate Products and Sales
for book in books:
    title, publisher, published_date, categories_raw = book

    # Clean category list
    try:
        categories = ", ".join(ast.literal_eval(categories_raw)) if categories_raw else "General"
    except Exception:
        categories = categories_raw or "General"

    # Random product values
    price = round(random.uniform(8.99, 29.99), 2)
    stock_quantity = random.randint(50, 500)
    is_active = True

    # Insert into Products
    cursor.execute('''
        INSERT INTO Products (name, category, brand, release_date, price, stock_quantity, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (title, categories, publisher, published_date, price, stock_quantity, is_active))
    
    product_id = cursor.lastrowid

    # Generate monthly sales from Jan 2023 to Apr 2025
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 4, 1)
    current_date = start_date

    while current_date <= end_date:
        units_sold = max(0, int(random.gauss(100, 40)))
        store_location = random.choice(stores)
        discount = round(random.uniform(0, 0.3), 2)  # Up to 30% discount

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