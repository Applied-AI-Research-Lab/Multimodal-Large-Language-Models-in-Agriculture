import os
import pandas as pd
import shutil
from PIL import Image

"""
Create dataset from image folders
"""


def rename_images_and_create_csv(dataset_path, output_csv):
    image_data = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)

        if os.path.isdir(category_path):
            for idx, image_file in enumerate(os.listdir(category_path)):
                ext = os.path.splitext(image_file)[1]
                unique_filename = f"{category}-{idx}{ext}"
                old_path = os.path.join(category_path, image_file)
                new_path = os.path.join(category_path, unique_filename)

                os.rename(old_path, new_path)
                image_data.append([unique_filename, category])

    df = pd.DataFrame(image_data, columns=['Image', 'Category'])
    df.to_csv(output_csv, index=False)
    print(f'CSV file saved: {output_csv}')


'''
Resize images in folder keeping the scale
'''


def resize_images_in_folder(src_folder, dest_folder, width):
    os.makedirs(dest_folder, exist_ok=True)

    for root, _, files in os.walk(src_folder):
        relative_path = os.path.relpath(root, src_folder)
        dest_subfolder = os.path.join(dest_folder, relative_path)
        os.makedirs(dest_subfolder, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_subfolder, file)

            try:
                with Image.open(src_file) as img:
                    aspect_ratio = img.height / img.width
                    new_height = int(width * aspect_ratio)
                    img_resized = img.resize((width, new_height), Image.LANCZOS)
                    img_resized.save(dest_file)
            except Exception as e:
                print(f"Error processing {src_file}: {e}")


'''
Create sub-folders by size
Resize images in folders keeping the scale
'''


def process_apple_dataset(src_root, dest_root):
    sizes = [256, 150, 100, 50]

    for size in sizes:
        dest_folder = os.path.join(dest_root, str(size))
        if size == 256:
            shutil.copytree(src_root, dest_folder, dirs_exist_ok=True)
        else:
            resize_images_in_folder(src_root, dest_folder, size)

# Create dataset from image folders
# rename_images_and_create_csv('../Datasets/Apple/', '../Datasets/apple_data.csv')
# rename_images_and_create_csv('../Datasets/Corn/', '../Datasets/corn_data.csv')

# Create sub-folders by size
# Resize images in folders keeping the scale
# process_apple_dataset("../Datasets/Apple/original", "../Datasets/AppleNew")
# process_apple_dataset("../Datasets/Corn/original", "../Datasets/CornNew")
