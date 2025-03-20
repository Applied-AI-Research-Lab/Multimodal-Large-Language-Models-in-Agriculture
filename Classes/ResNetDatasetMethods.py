import pandas as pd

class ResNetDataProcessor:
    def __init__(self, file_path, base_url):
        """
        Initialize the processor with the CSV file path and base URL.
        :param file_path: Path to the CSV file
        :param base_url: Base URL for constructing image URLs
        """
        self.file_path = file_path
        self.base_url = base_url
        self.data = None

    def load_data(self):
        """
        Load the data from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"File not found at {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def update_image_urls(self):
        """
        Update the 'Image' column with new URLs based on Category and Image name.
        """
        if self.data is not None:
            try:
                if 'Category' in self.data.columns and 'Image' in self.data.columns:
                    self.data['Image'] = self.data.apply(
                        lambda row: f"{self.base_url}{row['Category']}/{row['Image']}", axis=1
                    )
                    print("Image URLs updated successfully.")
                else:
                    print("Columns 'Category' or 'Image' not found in the data.")
            except Exception as e:
                print(f"Error updating image URLs: {e}")
        else:
            print("Data not loaded. Cannot update image URLs.")

    def save_data(self, output_path):
        """
        Save the updated data to a new CSV file.
        :param output_path: Path to save the updated CSV file
        """
        if self.data is not None:
            try:
                self.data.to_csv(output_path, index=False)
                print(f"Updated data saved to {output_path}")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("Data not loaded. Cannot save data.")

# type = 'Apple'
type = 'Corn'

# Creating train_set with 100px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/train_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/100/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_train_set_100.csv')

# Creating train_set with 150px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/train_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/150/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_train_set_150.csv')

# Creating train_set with 256px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/train_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/256/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_train_set_256.csv')

# Creating validation_set with 100px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/validation_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/100/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_validation_set_100.csv')

# Creating validation_set with 150px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/validation_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/150/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_validation_set_150.csv')

# Creating validation_set with 256px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/validation_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/256/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_validation_set_256.csv')

# Creating test_set with 100px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/test_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/100/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_test_set_100.csv')

# Creating test_set with 150px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/test_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/150/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_test_set_150.csv')

# Creating test_set with 256px images urls
# processor = ResNetDataProcessor('../Datasets/'+type+'/test_set.csv', 'https://applied-ai.gr/projects/agriculture/'+type+'/256/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../ResNet/Datasets/'+type+'_test_set_256.csv')