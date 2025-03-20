import os
import re
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import random
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold


class DatasetMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.pre_path = '../Datasets/'

    def just_read_csv(self):
        return pd.read_csv(self.pre_path + self.dataset_path)

    """
    If the 'id' column is missing, create a new column named 'id' starting from 1
    """

    def add_id_column(self):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Add a new column 'id' with sequential IDs starting from 1
        df.insert(0, 'id', range(1, len(df) + 1))

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_before_id.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame back to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

    """
    The method takes an array of column names (column_names) as input and removes empty rows.
    If the array is empty, each and every column is checked.
    Caution! The original dataset will be renamed to _original1,
         while the most current dataset will take the name of the original dataset
    """

    def remove_rows_with_empty_fields(self, column_names):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # If column_names is empty, check for empty fields in all columns
        if not column_names:
            # Check for empty fields in all columns and remove corresponding rows
            df = df.dropna(how='any')
        else:
            # Check for empty fields in specified columns and remove corresponding rows
            df = df.dropna(subset=column_names, how='any')

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original1.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": 'Empty rows removed'}

    """    
    The method takes an array of column names (columns_to_remove) as input and removes them entirely.
    Caution! The original dataset will be renamed to _original2,
         while the most current dataset will take the name of the original dataset
    """

    def remove_columns_and_save(self, columns_to_remove):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Automatically remove 'Unnamed: 0' column if present
        if 'Unnamed: 0' in df.columns:
            columns_to_remove.append('Unnamed: 0')

        # Remove the specified columns
        df = df.drop(columns=columns_to_remove, errors='ignore')

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original2' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original2.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": 'The specified columns have been removed'}

    """
    Display the unique labels in a specific column (column_name)
    """

    def display_unique_values(self, column_name):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Get the unique values and their counts
        unique_values_counts = df[column_name].value_counts()

        print(f"Unique values in column '{column_name}' ({len(unique_values_counts)}):")
        for value, count in unique_values_counts.items():
            print(f"Label: {value}: Count: {count}")

    """
    This method creates a subset (total_rows) of the original dataset,
    ensuring the appropriate distribution of the (stratified_column) values
    Caution! The original dataset will be renamed to _original5,
         while the most current dataset will take the name of the original dataset
    """

    def create_stratified_subset(self, total_rows, stratified_column):
        # Load the dataset
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Check the unique values in the stratified column
        unique_values = df[stratified_column].unique()

        # Create an empty DataFrame to store the subset
        subset_df = pd.DataFrame()

        # Define the number of rows you want for each value in the stratified column
        rows_per_value = total_rows // len(unique_values)

        # Loop through each unique value and sample rows
        for value in unique_values:
            value_subset = df[df[stratified_column] == value].sample(rows_per_value, random_state=42)
            subset_df = pd.concat([subset_df, value_subset])

        # If the total number of rows is less than the specified total, sample the remaining rows from the entire dataset
        remaining_rows = total_rows - len(subset_df)
        remaining_subset = df.sample(remaining_rows, random_state=42)
        subset_df = pd.concat([subset_df, remaining_subset])

        # Optionally, you can shuffle the final subset
        subset_df = subset_df.sample(frac=1, random_state=42)

        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original5.csv'
        os.rename(self.pre_path + self.dataset_path, original_file_path)

        # Save the modified DataFrame to a new CSV file
        subset_df.to_csv(self.pre_path + self.dataset_path, index=False)

        return {"status": True, "data": "Subset created"}

    """
    Split the dataset into train, validation, and test sets.
    By providing the stratify_column argument, the stratify function ensures that
    the distribution of labels or classes is maintained in both sets.
    """

    def split_dataset(self, stratify_column=''):
        train_file_path = 'train_set.csv'
        valid_file_path = 'validation_set.csv'
        test_file_path = 'test_set.csv'

        df = pd.read_csv(self.pre_path + self.dataset_path, on_bad_lines='skip')  # Read the cleaned dataset CSV file

        # Split the dataset into train, validation, and test sets while stratifying by the stratify_column
        if stratify_column:  # If stratify_column is provided, then stratify
            train_valid, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[stratify_column])
            train, valid = train_test_split(train_valid, test_size=0.2, random_state=42,
                                            stratify=train_valid[stratify_column])
        else:  # Split the dataset without stratifying
            train_valid, test = train_test_split(df, test_size=0.2, random_state=42)
            train, valid = train_test_split(train_valid, test_size=0.2, random_state=42)

        # Save the split datasets to separate CSV files
        train.to_csv(self.pre_path + train_file_path, index=False)
        valid.to_csv(self.pre_path + valid_file_path, index=False)
        test.to_csv(self.pre_path + test_file_path, index=False)

        return {"status": True, "data": "Splitting succeed"}

    def save_unique_categories_to_json(self, csv_file_path, json_file_path):
        """
        Extract unique categories from the 'Category' column of a CSV file and save them to a JSON file.

        Args:
            csv_file_path (str): Path to the input CSV file.
            json_file_path (str): Path to the output JSON file.
        """
        # Load the CSV file
        df = pd.read_csv(self.pre_path + csv_file_path)

        # Extract unique categories
        unique_categories = df['Category'].dropna().unique()

        # Sort the categories (optional, for consistent ordering)
        sorted_categories = sorted(unique_categories)

        # Create the dictionary in the specified format
        categories_dict = {"categories": sorted_categories}

        # Write the dictionary to a JSON file
        with open(self.pre_path + json_file_path, 'w') as json_file:
            json.dump(categories_dict, json_file, indent=4)

        print(f"Unique categories have been saved to {json_file_path}")

    """
    Further randomly split the train_set for Progressive Fine-Tuning
    """

    def split_train_set(self, label, input_file, output_prefix, num_splits=4):
        df = pd.read_csv(input_file)

        # Get the stratification column
        y = df[label]

        # Initialize the splitter
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

        # Generate the splits
        output_files = []

        for i, (_, split_idx) in enumerate(skf.split(df, y)):
            # Create the output file name
            output_file = f"{output_prefix}_{i + 1}.csv"

            # Extract the subset
            split_df = df.iloc[split_idx]

            # Save to CSV
            split_df.to_csv(output_file, index=False)

            output_files.append(output_file)

            print(f"Split {i + 1}: {len(split_df)} rows, " + label + " distribution:")
            print(split_df[label].value_counts(normalize=True))

        return output_files

# Instantiate the DatasetMethods class by providing the (dataset_path)
# type = 'apple'
# type = 'corn'
# DTS = DatasetMethods(dataset_path=type.title() + '/' + type + '_data.csv')

# Read the csv
# print(DTS.just_read_csv())

# Identify the unique labels in a specific column (column_name) to understand your dataset
# DTS.display_unique_values(column_name='Category')

# If the 'id' column is missing, create a new column named 'id' starting from 1
# DTS.add_id_column()

# Obtain a subset of the dataset with a specific number of rows (total_rows),
# while ensuring appropriate label distribution by stratifying a specific column (stratified_column)
# DTS.create_stratified_subset(800, 'Category')

# Split the dataset into training, validation, and test sets.
# Provide the column name (stratify_column) as an argument if you need to control the distribution
# print(DTS.split_dataset(stratify_column='Category'))

# Extract unique categories from the 'Category' column of a CSV file and save them to a JSON file.
# DTS.save_unique_categories_to_json(type.title() + '/test_set.csv', 'categories-' + type + '.json')

# Check distributions
# DTS2 = DatasetMethods(dataset_path=type.title() + '/test_set.csv')
# DTS2 = DatasetMethods(dataset_path=type.title() + '/validation_set.csv')
# DTS2 = DatasetMethods(dataset_path=type.title() + '/train_set.csv')
# DTS2.display_unique_values(column_name='Category')

# Further randomly split the train_set for Progressive Fine-Tuning
# DTS.split_train_set('Category', "../Datasets/" + type.title() + "/train_set.csv",
#                     "../Datasets/" + type.title() + "/train_set", num_splits=4)
