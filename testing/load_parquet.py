import pandas as pd
import os

# Define the path to the parquet files
data_folder = './data/information_operations'

# Load all parquet files in the folder
data_frames = []
for file_name in os.listdir(data_folder):
    if file_name.endswith('.parquet'):
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_parquet(file_path)
        data_frames.append(df)

# Concatenate all data frames into one
combined_df = pd.concat(data_frames, ignore_index=True)

# Display information about the dataframe
print("Column names:", combined_df.columns.tolist())
print("Number of entries:", len(combined_df))
print("DataFrame info:")
print(combined_df.info())

print(combined_df.keys())

print(combined_df['post_text'].head())