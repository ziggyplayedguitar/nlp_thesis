import pandas as pd
from pathlib import Path

# Load the parquet file
parquet_path = Path('data/information_operations/Russia_2_part_1.gzip.parquet')
df = pd.read_parquet(parquet_path)

# Display basic information
print("\nColumns in the dataset:")
print(df.columns.tolist())

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nData types of each column:")
print(df.dtypes)

print("\nNumber of rows:", len(df))

# Display some example posts
print("\nExample posts:")
for i, post in enumerate(df['post_text'].head(5)):
    print(f"\nExample {i+1}:")
    print(post) 