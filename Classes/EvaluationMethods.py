import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid PyCharm compatibility issues
import matplotlib.pyplot as plt
import os
import seaborn as sns


class EvaluationMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.pre_path = '../Datasets/'

    def evaluate_results(self, original, prediction, model_name):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        accuracy = round(accuracy_score(data[original], data[prediction]), 4)
        precision = round(precision_score(data[original], data[prediction], average='weighted'), 4)
        recall = round(recall_score(data[original], data[prediction], average='weighted'), 4)
        f1 = round(f1_score(data[original], data[prediction], average='weighted'), 4)

        # Create a DataFrame with the evaluation results including the 'model' column
        evaluation_df = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1]
        })

        # Append the results to the existing CSV file or create a new one
        evaluation_df.to_csv(self.pre_path + 'evaluation-results.csv', mode='a',
                             header=not os.path.exists(self.pre_path + 'evaluation-results.csv'), index=False)

        return {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

    def scatterplot(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)
        prediction = df[prediction_column]
        original = df[original_column]

        # Calculate Mean Absolute Error
        mae = abs(original - prediction).mean()

        # Create a scatter plot with a regression line
        sns.regplot(x=original, y=prediction, scatter_kws={'alpha': 0.5})

        plt.xlabel(original_column)
        plt.ylabel(prediction_column)

        # Save the scatterplot image to the Datasets folder
        plt.savefig(os.path.join(self.pre_path + 'Plots/', prediction_column + '.png'))

        # Show the plot
        plt.show()

        return mae

    def count_matching_rows(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Count the number of same value rows
        matching_rows = df[df[original_column] == df[prediction_column]]

        return len(matching_rows)

    def plot_histograms(self, original_column, prediction_column):
        dataframe = pd.read_csv(self.pre_path + self.dataset_path)

        # Separate predicted probabilities by class
        predicted_probabilities_class_0 = dataframe.loc[dataframe[original_column] == 0, prediction_column]
        predicted_probabilities_class_1 = dataframe.loc[dataframe[original_column] == 1, prediction_column]

        # Plot histograms
        plt.figure(figsize=(10, 5))

        # Histogram for class 0
        plt.subplot(1, 2, 1)
        plt.hist(predicted_probabilities_class_0, bins=20, color='blue', alpha=0.7)
        plt.title('Predicted Probabilities - Class 0')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')

        # Histogram for class 1
        plt.subplot(1, 2, 2)
        plt.hist(predicted_probabilities_class_1, bins=20, color='orange', alpha=0.7)
        plt.title('Predicted Probabilities - Class 1')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, original_column, prediction_column):
        dataframe = pd.read_csv(self.pre_path + self.dataset_path)

        # Extract data from DataFrame
        y_true = dataframe[original_column]
        y_pred = dataframe[prediction_column]

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix \n('+prediction_column+')')
        plt.show()

    """
    Plot a stacked bar chart showing the distribution of labels across categories in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """

    def plot_stacked_bar_chart(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        cross_tab = pd.crosstab(data[original_column], data[prediction_column])
        # Calculate row-wise percentages
        cross_tab_percent = cross_tab.apply(lambda x: x * 100 / x.sum(), axis=1)

        # Plotting the stacked bar chart
        ax = cross_tab_percent.plot(kind='bar', stacked=True, figsize=(10, 6))

        # Adding labels and title
        plt.title(f'Stacked Bar Chart of {original_column} vs. {prediction_column}')
        plt.xlabel(original_column)
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)

        # Adding percentages as text on each bar segment
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center', fontsize=8)

        plt.show()

    """
    Plot a grouped bar chart showing the relationship between labels in two columns.

    Args:
    column1 (str): The name of the first column with string labels.
    column2 (str): The name of the second column with string labels.
    """
    def plot_grouped_bar_chart(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        pivot_table = data.groupby([original_column, prediction_column]).size().unstack(fill_value=0)
        pivot_table.plot(kind='bar', figsize=(10, 6))
        plt.title(f'Relationship between {original_column} and {prediction_column}')
        plt.xlabel(original_column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    """
        Plot a heatmap showing relationships and patterns between label categories in two columns.

        Args:
        column1 (str): The name of the first column with string labels.
        column2 (str): The name of the second column with string labels.
    """
    def plot_heatmap(self, original_column, prediction_column):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        cross_tab = pd.crosstab(data[original_column], data[prediction_column])

        # Set larger font sizes for all text elements
        plt.rcParams.update({
            'font.size': 20,  # Base font size
            'axes.titlesize': 20,  # Title font size
            'axes.labelsize': 20,  # X and Y label font size
            'xtick.labelsize': 20,  # X tick label size
            'ytick.labelsize': 20,  # Y tick label size
            'legend.fontsize': 20,  # Legend font size
            'figure.titlesize': 20  # Figure title size
        })

        plt.figure(figsize=(12, 10))

        # Increase annotation font size in the heatmap
        ax = sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu',
                         annot_kws={"size": 20})  # Larger font for the numbers in cells

        # plt.title(f'Heatmap of {original_column} vs. {prediction_column}', fontsize=20, pad=20)
        plt.xlabel(prediction_column, fontsize=20, labelpad=10)
        plt.ylabel(original_column, fontsize=20, labelpad=10)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        output_dir = "../Plots"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'heatmap_{original_column}_vs_{prediction_column}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        # Instead of plt.show(), which causes the error
        # plt.show()

        print(f"Heatmap saved at: {output_path}")

        # Close the figure to free memory
        plt.close()


# type = "Apple"
type = "Corn"

# Instantiate the DatasetMethods class by providing the (dataset_path)
EVM = EvaluationMethods(dataset_path='../Datasets/' + type + '/test_set.csv')
# EVM = EvaluationMethods(dataset_path='../ResNet/Datasets/'+type.title()+'-test_set_256_with_predictions.csv')
# EVM = EvaluationMethods(dataset_path='../ResNet/Datasets/'+type.title()+'-test_set_150_with_predictions.csv')
# EVM = EvaluationMethods(dataset_path='../ResNet/Datasets/'+type.title()+'-test_set_100_with_predictions.csv')

# # Evaluate the predictions made by each model
# print(f'GPT-4o-Resolution-256: ' + str(EVM.evaluate_results('Category', 'GPT-4o-Resolution-256', 'GPT-4o-Resolution-256')))
# print(f'GPT-4o-mini-Resolution-256: ' + str(EVM.evaluate_results('Category', 'GPT-4o-mini-Resolution-256', 'GPT-4o-mini-Resolution-256')))
# print(f'Phase-1-Resolution-256: ' + str(EVM.evaluate_results('Category', 'Phase-1-Resolution-256', 'Phase-1-Resolution-256')))
# print(f'Phase-2-Resolution-256: ' + str(EVM.evaluate_results('Category', 'Phase-2-Resolution-256', 'Phase-2-Resolution-256')))
# print(f'Phase-3-Resolution-256: ' + str(EVM.evaluate_results('Category', 'Phase-3-Resolution-256', 'Phase-3-Resolution-256')))
# print(f'Phase-4-Resolution-256: ' + str(EVM.evaluate_results('Category', 'Phase-4-Resolution-256', 'Phase-4-Resolution-256')))
# print(f'ResNet50-256-Predictions: ' + str(EVM.evaluate_results('Category', 'ResNet50-Predictions', 'ResNet50-Predictions')))

# print(f'GPT-4o-Resolution-150: ' + str(EVM.evaluate_results('Category', 'GPT-4o-Resolution-150', 'GPT-4o-Resolution-150')))
# print(f'GPT-4o-mini-Resolution-150: ' + str(EVM.evaluate_results('Category', 'GPT-4o-mini-Resolution-150', 'GPT-4o-mini-Resolution-150')))
# print(f'Phase-1-Resolution-150: ' + str(EVM.evaluate_results('Category', 'Phase-1-Resolution-150', 'Phase-1-Resolution-150')))
# print(f'Phase-2-Resolution-150: ' + str(EVM.evaluate_results('Category', 'Phase-2-Resolution-150', 'Phase-2-Resolution-150')))
# print(f'Phase-3-Resolution-150: ' + str(EVM.evaluate_results('Category', 'Phase-3-Resolution-150', 'Phase-3-Resolution-150')))
# print(f'Phase-4-Resolution-150: ' + str(EVM.evaluate_results('Category', 'Phase-4-Resolution-150', 'Phase-4-Resolution-150')))
# print(f'ResNet50-150-Predictions: ' + str(EVM.evaluate_results('Category', 'ResNet50-Predictions', 'ResNet50-Predictions')))

# print(f'GPT-4o-Resolution-100: ' + str(EVM.evaluate_results('Category', 'GPT-4o-Resolution-100', 'GPT-4o-Resolution-100')))
# print(f'GPT-4o-mini-Resolution-100: ' + str(EVM.evaluate_results('Category', 'GPT-4o-mini-Resolution-100', 'GPT-4o-mini-Resolution-100')))
# print(f'Phase-1-Resolution-100: ' + str(EVM.evaluate_results('Category', 'Phase-1-Resolution-100', 'Phase-1-Resolution-100')))
# print(f'Phase-2-Resolution-100: ' + str(EVM.evaluate_results('Category', 'Phase-2-Resolution-100', 'Phase-2-Resolution-100')))
# print(f'Phase-3-Resolution-100: ' + str(EVM.evaluate_results('Category', 'Phase-3-Resolution-100', 'Phase-3-Resolution-100')))
# print(f'Phase-4-Resolution-100: ' + str(EVM.evaluate_results('Category', 'Phase-4-Resolution-100', 'Phase-4-Resolution-100')))
# print(f'ResNet50-100-Predictions: ' + str(EVM.evaluate_results('Category', 'ResNet50-Predictions', 'ResNet50-Predictions')))

# # # Best model at each resolution making predictions on a different crop to evaluate generalization
# print(f'Best-Apple-Trained-Model-Predictions-on-Corns-256: ' + str(EVM.evaluate_results('Category', 'Best-Apple-Trained-Model-Predictions-on-Corns-256', 'Best-Apple-Trained-Model-Predictions-on-Corns-256')))
# print(f'Best-Apple-Trained-Model-Predictions-on-Corns-150: ' + str(EVM.evaluate_results('Category', 'Best-Apple-Trained-Model-Predictions-on-Corns-150', 'Best-Apple-Trained-Model-Predictions-on-Corns-150')))
# print(f'Best-Apple-Trained-Model-Predictions-on-Corns-100: ' + str(EVM.evaluate_results('Category', 'Best-Apple-Trained-Model-Predictions-on-Corns-100', 'Best-Apple-Trained-Model-Predictions-on-Corns-100')))
# print(f'Best-Corn-Trained-Model-Predictions-on-Apples-256: ' + str(EVM.evaluate_results('Category', 'Best-Corn-Trained-Model-Predictions-on-Apples-256', 'Best-Corn-Trained-Model-Predictions-on-Apples-256')))
# print(f'Best-Corn-Trained-Model-Predictions-on-Apples-150: ' + str(EVM.evaluate_results('Category', 'Best-Corn-Trained-Model-Predictions-on-Apples-150', 'Best-Corn-Trained-Model-Predictions-on-Apples-150')))
# print(f'Best-Corn-Trained-Model-Predictions-on-Apples-100: ' + str(EVM.evaluate_results('Category', 'Best-Corn-Trained-Model-Predictions-on-Apples-100', 'Best-Corn-Trained-Model-Predictions-on-Apples-100')))

# # # Cross-resolution evaluation: Assessing models trained on higher resolutions making predictions on lower
# # # resolutions, and vice versa
# print(f'Apple-High-to-Low-Res-Trained-256-Prediction-100: ' + str(EVM.evaluate_results('Category', 'Apple-High-to-Low-Res-Trained-256-Prediction-100', 'Apple-High-to-Low-Res-Trained-256-Prediction-100')))
# print(f'Apple-Low-to-High-Res-Trained-100-Prediction-256: ' + str(EVM.evaluate_results('Category', 'Apple-Low-to-High-Res-Trained-100-Prediction-256', 'Apple-Low-to-High-Res-Trained-100-Prediction-256')))
# print(f'Corn-High-to-Low-Res-Trained-256-Prediction-100: ' + str(EVM.evaluate_results('Category', 'Corn-High-to-Low-Res-Trained-256-Prediction-100', 'Corn-High-to-Low-Res-Trained-256-Prediction-100')))
# print(f'Corn-Low-to-High-Res-Trained-100-Prediction-256: ' + str(EVM.evaluate_results('Category', 'Corn-Low-to-High-Res-Trained-100-Prediction-256', 'Corn-Low-to-High-Res-Trained-100-Prediction-256')))

# # # Evaluate predictions on fine-tuned models trained on the full training set
# print(f'GPT-Resolution-256: ' + str(EVM.evaluate_results('Category', 'GPT-Resolution-256', 'GPT-Resolution-256')))
# print(f'GPT-Resolution-150: ' + str(EVM.evaluate_results('Category', 'GPT-Resolution-150', 'GPT-Resolution-150')))
# print(f'GPT-Resolution-100: ' + str(EVM.evaluate_results('Category', 'GPT-Resolution-100', 'GPT-Resolution-100')))

# # # ResNet-50 Predictions
# print(f'ResNet50-Predictions-256-Bayesian-Optimization: ' + str(EVM.evaluate_results('Category', 'ResNet50-Predictions-256-Bayesian-Optimization', 'ResNet50-Predictions-256-Bayesian-Optimization')))
# print(f'ResNet50-Predictions-Bayesian-Optimization: ' + str(EVM.evaluate_results('Category', 'ResNet50-Predictions-Bayesian-Optimization', 'ResNet50-Predictions-Bayesian-Optimization')))
# print(f'ResNet50-Predictions-Bayesian-Optimization: ' + str(EVM.evaluate_results('Category', 'ResNet50-Predictions-Bayesian-Optimization', 'ResNet50-Predictions-Bayesian-Optimization')))

# # # Heatmaps
# print(EVM.plot_heatmap(original_column='Category', prediction_column='GPT-Resolution-256'))
# print(EVM.plot_heatmap(original_column='Category', prediction_column='GPT-4o-Resolution-256'))
# print(EVM.plot_heatmap(original_column='Category', prediction_column='ResNet50-Predictions-256-Bayesian-Optimization'))

