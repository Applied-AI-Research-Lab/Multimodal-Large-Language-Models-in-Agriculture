import os
import pandas as pd
import openai
from openai import OpenAI
import json
import logging
import re
import time


class GPTmethods:
    def __init__(self, params):
        """
        Initialize the class with the provided parameters.
        The constructor sets up the OpenAI API key, model configuration, and various other
        parameters needed for generating prompts and making predictions.

        Args:
            params (dict): A dictionary containing the configuration settings.
        """
        # Access the OpenAI API key from environment variables
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        # Initialize class variables using the provided parameters
        self.model_id = params['model_id']  # The model ID to use (e.g., gpt-4o)
        self.prediction_column = params['prediction_column']  # Specifies the column where predictions will be stored
        self.pre_path = params['pre_path']  # The path to datasets
        self.data_set = params['data_set']  # Defines the path to the CSV dataset file
        self.prompt_array = params['prompt_array']  # A dictionary with additional data
        self.system = params['system']  # System-level message for context in the conversation
        self.prompt = params['prompt']  # The base prompt template
        self.feature_col = params['feature_col']  # Column name for feature input
        self.label_col = params['label_col']  # Column name for the label
        self.json_key = params['json_key']  # Key for extracting relevant data from the model's response
        self.max_tokens = params['max_tokens']  # Maximum number of tokens to generate in the response
        self.temperature = params['temperature']  # Controls response randomness (0 is most deterministic)
        self.path_to_image = params['path_to_image']  # Only for specific cases

    """
    Generates a custom prompt
    """

    def generate_prompt(self, feature):
        # Read the JSON file
        with open(self.pre_path + self.prompt_array['json_file'], 'r') as file:
            data = json.load(file)

        # Create a new dictionary with the product title and existing categories
        new_data = {
            # "product_title": feature,
            "categories": data["categories"]
        }

        # Convert the dictionary to a JSON-formatted string
        replacement = json.dumps(new_data, indent=2)

        updated_prompt = self.prompt.replace('[json]', replacement)

        # If the prompt is simple you can avoid this method by setting updated_prompt = self.prompt + feature
        return updated_prompt  # This method returns the whole new custom prompt

    """
    Creates a training and validation JSONL file for GPT fine-tuning.
    The method reads a CSV dataset, generates prompt-completion pairs for each row, and formats the data into
    the required JSONL structure for GPT fine-tuning.
    The generated JSONL file will contain system, user, and assistant messages for each training || validation instance.
    """

    def create_jsonl(self, data_type, data_set, parent_path, save_path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + data_set)
        data = []  # List to store the formatted data for each row

        # Iterate over each row in the DataFrame to format the data for fine-tuning
        for index, row in df.iterrows():
            data.append(
                {
                    "messages": [

                        {
                            "role": "user",
                            "content": self.generate_prompt(feature=row[self.feature_col])  # Generate user prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": parent_path + str(row['Category']) + '/' + str(row['Image'])
                                    }
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": f"{{\"{self.json_key}\": \"{row[self.label_col]}\"}}"  # Assistant's response
                        }
                    ]
                }
            )

        # Define the output file path for the JSONL file
        output_file_path = self.pre_path + save_path + "ft_dataset_gpt_" + data_type + ".jsonl"  # Define the path
        # Write the formatted data to the JSONL file
        with open(output_file_path, 'w') as output_file:
            for record in data:
                # Convert each dictionary record to a JSON string and write it to the file
                json_record = json.dumps(record)
                output_file.write(json_record + '\n')

        # Return a success message with the file path
        return {"status": True, "data": f"JSONL file '{output_file_path}' has been created."}

    """
    Create a conversation with the GPT model by sending a series of messages and receiving a response.
    This method constructs the conversation and returns the model's reply based on the provided messages.
    """

    def gpt_conversation(self, conversation):
        # Instantiate the OpenAI client to interact with the GPT model
        client = OpenAI()
        # Send the conversation to the model and get the response
        completion = client.chat.completions.create(
            model=self.model_id,  # Specify the model to use for the conversation
            messages=conversation  # Pass the conversation history as input
        )
        # Return the message from the model's response
        return completion.choices[0].message

    """
    Cleans the response from the GPT model by attempting to extract and parse a JSON string.
    If the response is already in dictionary format, it is returned directly.
    If the response contains a JSON string, it will be extracted, cleaned, and parsed.
    If no valid JSON is found or a decoding error occurs, an error message is logged.
    """

    def clean_response(self, response, a_field):
        if isinstance(response, dict):
            return {"status": True, "data": response}
        try:
            start_index = response.find('{')
            end_index = response.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = response[start_index:end_index + 1]
                # Replace single quotes with double quotes
                json_str = re.sub(r"'", '"', json_str)
                # Handle missing quotes around keys
                json_str = re.sub(r'([a-zA-Z0-9_]+):', r'"\1":', json_str)
                # Modified regex to handle multi-word values
                json_str = re.sub(r':\s*([a-zA-Z0-9_\s]+)([\s,}\]])', r': "\1"\2', json_str)
                # Clean up any double spaces in the values
                json_str = re.sub(r'\s+', ' ', json_str)
                json_data = json.loads(json_str)
                return {"status": True, "data": json_data}
            else:
                logging.error(f"No JSON found in the response. The input '{a_field}', resulted in the "
                              f"following response: {response}")
                return {
                    "status": False,
                    "data": f"No JSON found in the response. The input '{a_field}', "
                            f"resulted in the following response: {response}"
                }
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                          f"resulted in the following response: {response}")
            return {
                "status": False,
                "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                        f"resulted in the following response: {response}"
            }

    """
    Prompts the GPT model to generate a prediction based on the provided input.
    The method constructs a conversation with the model using the system message and user input, 
    and processes the model's response to return a clean, formatted prediction.
    """

    def gpt_prediction(self, input):
        conversation = []
        conversation.append({
            'role': 'user',
            "content": [
                {"type": "text", "text": self.generate_prompt(feature=input[self.feature_col])},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.path_to_image + str(input['Category']) + '/' + str(input['Image'])
                    }
                }
            ]
        })  # Generate the prompt
        # Instead of replacing url with id, you can use the feature column directly
        # in some cases "url": input[self.feature_col]

        # Get the model's response by passing the conversation to gpt_conversation
        conversation = self.gpt_conversation(conversation)
        # Extract the content of the GPT model's response
        content = conversation.content

        # Clean and format the response before returning it
        return self.clean_response(response=content, a_field=input[self.feature_col])

    """
    Makes predictions for a specific dataset and append the predictions to a new column.
    This method processes each row in the dataset, generates predictions using the GPT model, 
    and updates the dataset with the predicted values in the specified prediction column.
    """

    def predictions(self):

        # Start measuring time
        start_time = time.time()

        # Read the CSV dataset into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.data_set)

        # Create a copy of the original dataset (with '_original' appended to the filename)
        file_name_without_extension = os.path.splitext(os.path.basename(self.data_set))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original.csv'
        if not os.path.exists(original_file_path):
            os.rename(self.pre_path + self.data_set, original_file_path)

        # Check if the prediction_column is already present in the header
        if self.prediction_column not in df.columns:
            # If not, add the column to the DataFrame with pd.NA as the initial value
            df[self.prediction_column] = pd.NA

            # # Explicitly set the column type to a nullable integer
            # df = df.astype({prediction_column: 'Int64'})

        # Save the updated DataFrame back to CSV (if a new column is added)
        if self.prediction_column not in df.columns:
            df.to_csv(self.pre_path + self.data_set, index=False)

        # Set the dtype of the reason column to object
        # df = df.astype({reason_column: 'object'})

        # Iterate over each row in the DataFrame to make predictions
        for index, row in df.iterrows():
            # Make a prediction if the value in the prediction column is missing (NaN)
            if pd.isnull(row[self.prediction_column]):
                prediction = self.gpt_prediction(input=row)
                # If the prediction fails, log the error and break the loop
                if not prediction['status']:
                    print(prediction)
                    break
                else:
                    print(prediction)
                    # If the prediction data contains a valid value, update the DataFrame
                    if prediction['data'][self.json_key] != '':
                        # Update the CSV file with the new prediction values
                        df.at[index, self.prediction_column] = prediction['data'][self.json_key]
                        # for integers only
                        # df.at[index, prediction_column] = int(prediction['data'][self.json_key])

                        # Update the CSV file with the new values
                        df.to_csv(self.pre_path + self.data_set, index=False)
                    else:
                        logging.error(
                            f"No {self.json_key} instance was found within the data for '{row[self.feature_col]}', and the "
                            f"corresponding prediction response was: {prediction}.")
                        return {"status": False,
                                "data": f"No {self.json_key} instance was found within the data for '{row[self.feature_col]}', "
                                        f"and the corresponding prediction response was: {prediction}."}

                # break
            # Add a delay of 5 seconds (reduced for testing)

        # Change the column datatype after processing all predictions to handle 2.0 ratings
        # df[prediction_column] = df[prediction_column].astype('Int64')

        # End measuring time
        end_time = time.time()  # Record the end time

        # Calculate total time taken
        time_taken = end_time - start_time
        print(f"Total time taken for predictions: {time_taken:.2f} seconds")

        # After all predictions are made, return a success message
        return {"status": True, "data": 'Prediction have successfully been'}

    """
    Upload a dataset for GPT fine-tuning via the OpenAI API.
    The dataset file will be uploaded with the purpose of fine-tuning the model.
    """

    def upload_file(self, dataset):
        # Uploads the specified dataset file to OpenAI for fine-tuning.
        upload_file = openai.File.create(
            file=open(dataset, "rb"),  # Opens the dataset file in binary read mode
            purpose='fine-tune'  # Specifies the purpose of the upload as 'fine-tune'
        )
        return upload_file

    """
      Train the GPT model either through the API or by using the OpenAI UI for fine-tuning.
      Refer to the official OpenAI fine-tuning guide for more details: 
      https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model?ref=mlq.ai
      """

    def train_gpt(self, file_id):
        # Initiates a fine-tuning job using the OpenAI API with the provided training file ID and model ("gpt-4o").
        return openai.FineTuningJob.create(training_file=file_id, model="gpt-4o")
        # Optionally, check the status of the training job by calling:
        # openai.FineTuningJob.retrieve(file_id)

    """
    Delete a Fine-Tuned GPT model
    This method deletes a specified fine-tuned GPT model using OpenAI's API. 
    """

    def delete_finetuned_model(self, model):  # ex. model = ft:gpt-3.5-turbo-0613:personal::84kHoCN
        return openai.Model.delete(model)

    """
    Cancel Fine-Tuning Job
    This method cancels an ongoing fine-tuning job using OpenAI's API.
    """

    def cancel_gpt_finetuning(self, train_id):  # ex. id = ftjob-3C5lZD1ly5HHAleLwAqT7Qt
        return openai.FineTuningJob.cancel(train_id)

    """
    Retrieve All Fine-Tuned Models and Their Status
    This method fetches a list of fine-tuned models and their details using OpenAI's API. 
    The results include information such as the model IDs, statuses, and metadata.
    """

    def get_all_finetuned_models(self):
        return openai.FineTuningJob.list(limit=10)


# TODO: Before running the script:
#  Ensure the OPENAI_API_KEY is set as an environment variable to enable access to the OpenAI API.

"""
Configure the logging module to record error messages in a file named 'error_log.txt'.
"""
logging.basicConfig(filename='../error_log.txt', level=logging.ERROR)

"""
The `params` dictionary contains configuration settings for the AI model's prediction process. 
It includes specifications for the model ID, dataset details, system and task-specific prompts, 
and parameters for prediction output, response format, and model behavior.
"""
params = {
    'model_id': 'gpt-4o-mini',  # Specifies the GPT model ID for making predictions.
    'prediction_column': 'gpt_4o_mini_prediction',  # Specifies the column where predictions will be stored.
    'pre_path': '',  # Specifies the base directory path where dataset files are located.
    'data_set': 'test_set.csv',  # Defines the path to the CSV dataset file.
    'prompt_array': {},  # Can be an empty array for simple projects.
    # Defines the system prompt that describes the task.
    'system': 'You are an AI assistant specializing in image classification.',
    # Defines the prompt for the model, instructing it to make predictions and return its response in JSON format.
    # You can pass anything within brackets [example], which will be replaced during generate_prompt().
    'prompt': '',
    'feature_col': 'Image',  # Specifies the column in the dataset containing the text input/feature for predictions.
    'label_col': 'Category',  # Used only for creating training and validation prompt-completion pairs JSONL files.
    'json_key': 'category',  # Defines the key in the JSON response expected from the model, e.g. {"category": "value"}
    'max_tokens': 1000,  # Sets the maximum number of tokens the model should generate in its response.
    'temperature': 0,  # Sets the temperature for response variability; 0 provides the most deterministic response.
    'path_to_image': '',  # Only for specific cases
}

# type = 'apple'
type = 'corn'
params['pre_path'] = 'Datasets/' + type.title() + '/'
params['prompt_array'] = {'json_file': 'categories-' + type + '.json'}
params['prompt'] = 'Analyze the provided image of an ' + type + ' leaf using your computer vision capabilities. Classify the leaf into the most appropriate category based on its condition, choosing from the predefined list: [json] Provide your final classification in the following JSON format without explanations: {"category": "chosen_category_name"}'

# # # Apple:Resolution: 256px
# # gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # Phase 1 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-256:B9ojfjfp'
# params['prediction_column'] = 'Phase-1-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # Phase 2 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-256:B9rgDgVU'
# params['prediction_column'] = 'Phase-2-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # Phase 3 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-256:B9wnegyN'
# params['prediction_column'] = 'Phase-3-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # Phase 4 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-256:B9zes8zI'
# params['prediction_column'] = 'Phase-4-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'

# # # Apple:Resolution: 150px
# # gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'
# # gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'
# # Phase 1 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-150:BA0XO3aS'
# params['prediction_column'] = 'Phase-1-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'
# # Phase 2 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-150:BA1jLY4h'
# params['prediction_column'] = 'Phase-2-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'
# # Phase 3 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-150:BA6q0cyK'
# params['prediction_column'] = 'Phase-3-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'
# # Phase 4 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-150:BA7oD2kT'
# params['prediction_column'] = 'Phase-4-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'

# # # Apple:Resolution: 100px
# # gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'
# # gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'
# # Phase 1 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-100:BACKwp0N'
# params['prediction_column'] = 'Phase-1-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'
# # Phase 2 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-100:BAEfIBXZ'
# params['prediction_column'] = 'Phase-2-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'
# # Phase 3 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-100:BAJW3bri'
# params['prediction_column'] = 'Phase-3-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'
# # Phase 4 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-100:BAMVk7Cm'
# params['prediction_column'] = 'Phase-4-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'

# # # Corn:Resolution: 256px
# # gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'
# # gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'
# # Phase 1 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-256-corn:BAWXbcjw'
# params['prediction_column'] = 'Phase-1-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'
# # Phase 2 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-256-corn:BAXxy46P'
# params['prediction_column'] = 'Phase-2-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'
# # Phase 3 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-256-corn:BAZKgwpN'
# params['prediction_column'] = 'Phase-3-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'
# # Phase 4 - 256px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-256-corn:BAaQ3wfm'
# params['prediction_column'] = 'Phase-4-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'

# # # Corn:Resolution: 150px
# # gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'
# # gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'
# # Phase 1 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-150-corn:BAc2StJe'
# params['prediction_column'] = 'Phase-1-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'
# Phase 2 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-150-corn:BB8YhNDx'
# params['prediction_column'] = 'Phase-2-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'
# # Phase 3 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-150-corn:BB9ux708'
# params['prediction_column'] = 'Phase-3-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'
# # Phase 4 - 150px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-150-corn:BBAo9aAA'
# params['prediction_column'] = 'Phase-4-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'

# # # Corn:Resolution: 100px
# # gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'
# # gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'
# # Phase 1 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-100-corn:BBBljn0v'
# params['prediction_column'] = 'Phase-1-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'
# # Phase 2 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-100-corn:BBD8rd0v'
# params['prediction_column'] = 'Phase-2-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'
# # Phase 3 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-100-corn:BBDySt2J'
# params['prediction_column'] = 'Phase-3-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'
# # Phase 4 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-100-corn:BBF7QPJj'
# params['prediction_column'] = 'Phase-4-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'

# # # Best model at each resolution making predictions on a different crop to evaluate generalization
# # Best-Apple-Trained-Model-Predictions-on-Corns-256
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-256:B9wnegyN'
# params['prediction_column'] = 'Best-Apple-Trained-Model-Predictions-on-Corns-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'
# # Best-Apple-Trained-Model-Predictions-on-Corns-150
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-150:BA7oD2kT'
# params['prediction_column'] = 'Best-Apple-Trained-Model-Predictions-on-Corns-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'
# # Best-Apple-Trained-Model-Predictions-on-Corns-100
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-100:BAJW3bri'
# params['prediction_column'] = 'Best-Apple-Trained-Model-Predictions-on-Corns-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'
# # Best-Corn-Trained-Model-Predictions-on-Apples-256
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-256-corn:BAaQ3wfm'
# params['prediction_column'] = 'Best-Corn-Trained-Model-Predictions-on-Apples-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # Best-Corn-Trained-Model-Predictions-on-Apples-150
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-150-corn:BB9ux708'
# params['prediction_column'] = 'Best-Corn-Trained-Model-Predictions-on-Apples-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'
# # Best-Corn-Trained-Model-Predictions-on-Apples-100
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-100-corn:BBF7QPJj'
# params['prediction_column'] = 'Best-Corn-Trained-Model-Predictions-on-Apples-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'

# # # Cross-resolution evaluation: Assessing models trained on higher resolutions making predictions on lower
# # # resolutions, and vice versa
# # Apple-High-to-Low-Res-Trained-256-Prediction-100
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-256:B9wnegyN'
# params['prediction_column'] = 'Apple-High-to-Low-Res-Trained-256-Prediction-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'
# # Apple-Low-to-High-Res-Trained-100-Prediction-256
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-100:BAJW3bri'
# params['prediction_column'] = 'Apple-Low-to-High-Res-Trained-100-Prediction-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # Corn-High-to-Low-Res-Trained-256-Prediction-100
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-256-corn:BAaQ3wfm'
# params['prediction_column'] = 'Corn-High-to-Low-Res-Trained-256-Prediction-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'
# # Corn-Low-to-High-Res-Trained-100-Prediction-256
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-100-corn:BBF7QPJj'
# params['prediction_column'] = 'Corn-Low-to-High-Res-Trained-100-Prediction-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'

# # # Apple predictions on fine-tuned models trained on the full training set
# # Predictions on 256px images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:gpt-resolution-256:BCjDLKKo'
# params['prediction_column'] = 'GPT-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/256/'
# # Predictions on 150px images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:gpt-resolution-150:BCsfA4QN'
# params['prediction_column'] = 'GPT-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/150/'
# # Predictions on 100px images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:gpt-resolution-100:BCtaE9Ou'
# params['prediction_column'] = 'GPT-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Apple/100/'

# # # Corn predictions on fine-tuned models trained on the full training set
# # Predictions on 256px images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:gpt-resolution-256-corn:BCuf5MWW'
# params['prediction_column'] = 'GPT-Resolution-256'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/256/'
# # Predictions on 150px images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:gpt-resolution-150-corn:BD6xrnn3'
# params['prediction_column'] = 'GPT-Resolution-150'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/150/'
# # Predictions on 100px images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:gpt-resolution-100-corn:BD7xmXAh'
# params['prediction_column'] = 'GPT-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/agriculture/Corn/100/'

"""
Create an instance of the GPTmethods class, passing the `params` dictionary to the constructor for initialization.
"""
GPT = GPTmethods(params)

"""
Call the `predictions` method of the GPTmethods instance to make predictions on the specified dataset.
"""
GPT.predictions()

"""
Create JSONL training files for fine-tuning the GPT model
4 training sets for 3 different resolutions
"""
# parent_path = 'https://applied-ai.gr/projects/agriculture/' + type.title() + '/'

# 256px Width Size
# GPT.create_jsonl(data_type='train_1_256_size', data_set='train_set_1.csv', parent_path=parent_path + '256/', save_path='FineTuning/256/')
# GPT.create_jsonl(data_type='train_2_256_size', data_set='train_set_2.csv', parent_path=parent_path + '256/', save_path='FineTuning/256/')
# GPT.create_jsonl(data_type='train_3_256_size', data_set='train_set_3.csv', parent_path=parent_path + '256/', save_path='FineTuning/256/')
# GPT.create_jsonl(data_type='train_4_256_size', data_set='train_set_4.csv', parent_path=parent_path + '256/', save_path='FineTuning/256/')

# 150px Width Size
# GPT.create_jsonl(data_type='train_1_150_size', data_set='train_set_1.csv', parent_path=parent_path + '150/', save_path='FineTuning/150/')
# GPT.create_jsonl(data_type='train_2_150_size', data_set='train_set_2.csv', parent_path=parent_path + '150/', save_path='FineTuning/150/')
# GPT.create_jsonl(data_type='train_3_150_size', data_set='train_set_3.csv', parent_path=parent_path + '150/', save_path='FineTuning/150/')
# GPT.create_jsonl(data_type='train_4_150_size', data_set='train_set_4.csv', parent_path=parent_path + '150/', save_path='FineTuning/150/')

# 100px Width Size
# GPT.create_jsonl(data_type='train_1_100_size', data_set='train_set_1.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# GPT.create_jsonl(data_type='train_2_100_size', data_set='train_set_2.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# GPT.create_jsonl(data_type='train_3_100_size', data_set='train_set_3.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# GPT.create_jsonl(data_type='train_4_100_size', data_set='train_set_4.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')

"""
Create JSONL validation files for fine-tuning the GPT model
4 training sets for 3 different resolutions
"""
# 256px Width Size
# GPT.create_jsonl(data_type='validation_256_size', data_set='validation_set.csv', parent_path=parent_path + '256/', save_path='FineTuning/256/')
# 150px Width Size
# GPT.create_jsonl(data_type='validation_150_size', data_set='validation_set.csv', parent_path=parent_path + '150/', save_path='FineTuning/150/')
# 100px Width Size
# GPT.create_jsonl(data_type='validation_100_size', data_set='validation_set.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')

"""
Create JSONL training and validation files from the full dataset for fine-tuning the GPT model
1 training set for 3 different resolutions
"""
# parent_path = 'https://applied-ai.gr/projects/agriculture/' + type.title() + '/'
# GPT.create_jsonl(data_type='train_256_size', data_set='train_set.csv', parent_path=parent_path + '256/', save_path='FineTuning/256/')
# GPT.create_jsonl(data_type='train_150_size', data_set='train_set.csv', parent_path=parent_path + '150/', save_path='FineTuning/150/')
# GPT.create_jsonl(data_type='train_100_size', data_set='train_set.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# Validation files have already been generated.
