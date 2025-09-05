import polars as pl

file_path = "data/Data.csv"

# ----------- Laz loading for metadata -----------
lazy_df = pl.scan_csv(file_path)

# Preview
print("🔹 Dataset Preview:")
print(lazy_df.collect().heaD(5))  # replaces deprecated fetch()

# Schema (no expensive computation)
schema = lazy_df.collect_schema()
print("\n🔹 Number of features (columns):", len(schema))
print("🔹 Feature names:", list(schema.keys()))

# ----------- Null Values -----------
null_counts = (
    lazy_df.select([pl.col(col).null_count().alias(col) for col in schema.keys()])
    .collect()
    .transpose(include_header=True)
)
print("\n🔹 Null values per column:")
print(null_counts)

# ----------- Summary Statistics -----------
summary = lazy_df.describe()  # already returns DataFrame
print("\n🔹 Summary statistics:")
print(summary)

# ----------- Categorical Value Counts -----------
print("\n🔹 Example categorical counts (top 10 for each string column):")
for col, dtype in schema.items():
    if dtype == pl.Utf8:  # categorical-like
        print(f"\nColumn: {col}")
        counts = (
            lazy_df.group_by(col)
                   .agg(pl.len().alias("count"))
                   .sort("count", descending=True)
                   .limit(10)
                   .collect()
        )
        print(counts)

# ----------- Chunked Processing Example -----------
print("\n🔹 Chunked processing example (first 2 chunks):")
df_stream = pl.read_csv(file_path, low_memory=True)
chunk_size = 100_000

for i, chunk in enumerate(df_stream.iter_slices(n_rows=chunk_size)):
    print(f"\nChunk {i+1}: shape = {chunk.shape}")
    if "age" in chunk.columns:
        chunk = chunk.filter(pl.col("age") > 18)
    print(chunk.head(3))
    if i == 1:
        break

# ----------- Save Polars DataFrame -----------
df_stream.write_parquet("processed_Data.parquet")

print("\n✅ Data saved as processed_Data.parquet")
