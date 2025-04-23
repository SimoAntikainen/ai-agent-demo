from pathlib import Path
import polars as pl

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / 'book_reviews'

# Step 1: Load CSVs
schema_ratings = {
    "Id": pl.Utf8,  # original column name in CSV
    "Title": pl.Utf8,
    "Price": pl.Utf8,
    "User_id": pl.Utf8,
    "profileName": pl.Utf8,
    "review/helpfulness": pl.Utf8,
    "review/score": pl.Float64,
    "review/time": pl.Int64,
    "review/summary": pl.Utf8,
    "review/text": pl.Utf8,
}

schema_books_data = {
    "Title": pl.Utf8,
    "description": pl.Utf8,
    "authors": pl.Utf8,
    "image": pl.Utf8,
    "previewLink": pl.Utf8,
    "publisher": pl.Utf8,
    "publishedDate": pl.Utf8,
    "infoLink": pl.Utf8,
    "categories": pl.Utf8,
    "ratingsCount": pl.Float64,
}

ratings_df = pl.read_csv(DATA_DIR / "Books_rating.csv", schema_overrides=schema_ratings, low_memory=True)
book_data_df = pl.read_csv(DATA_DIR / "books_data.csv", schema_overrides=schema_books_data)

# Step 2: Count reviews per book Id
book_counts = (
    ratings_df
    .group_by("Id", "Title")
    .agg(pl.count("Id").alias("review_count"))
    .filter(pl.col("review_count") >= 1000)
    .rename({"Id": "productId"})  # Rename here early for consistency
)

# Step 3: Sample up to 100 book Ids
eligible_books = book_counts.sample(n=min(100, book_counts.height), seed=42)
selected_ids = eligible_books["productId"].to_list()
selected_titles = eligible_books["Title"].to_list()

# Step 4: Filter reviews to just those books
filtered_reviews = ratings_df.filter(pl.col("Id").is_in(selected_ids)).rename({"Id": "productId"})

# Step 5: Limit to 30 reviews per book — with some low ratings included
low_score_reviews = (
    filtered_reviews
    .filter(pl.col("review/score") <= 3.0)
    .group_by("productId", maintain_order=True)
    .head(10)
)

high_score_reviews = (
    filtered_reviews
    .filter(pl.col("review/score") > 3.0)
    .group_by("productId", maintain_order=True)
    .head(20)
)

# Combine both sets
balanced_reviews = pl.concat([low_score_reviews, high_score_reviews]).sort(["productId", "review/time"])

# Step 6: Filter book metadata
balanced_books = book_data_df.filter(pl.col("Title").is_in(selected_titles))

# Step 6.1: Add productId to balanced_books
balanced_books = balanced_books.join(
    eligible_books.select(["productId", "Title"]),
    on="Title",
    how="left"
)

# Step 7: Join authors and categories into reviews
balanced_reviews_with_meta = balanced_reviews.join(
    balanced_books.select(["Title", "authors", "categories"]),
    on="Title",
    how="left"
)

columns = ["productId"] + [col for col in balanced_books.columns if col != "productId"]
balanced_books = balanced_books.select(columns)

# Step 8: Save results
balanced_reviews_with_meta.write_csv(DATA_DIR / "balanced_reviews.csv")
balanced_books.write_csv(DATA_DIR / "balanced_books.csv")

print("✅ Saved balanced_reviews.csv with authors & categories and balanced_books.csv with productId")