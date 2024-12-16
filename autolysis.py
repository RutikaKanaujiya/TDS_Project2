import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
import chardet # type: ignore

# Load environment variables
load_dotenv()

# Set your AI Proxy token from environment variables
openai.api_key = os.getenv("AIPROXY_TOKEN")

# Function to detect the encoding of a file
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result["encoding"]

# Function to analyze and summarize the dataset
def analyze_dataset(data_file):
    try:
        # Detect file encoding
        encoding = detect_encoding(data_file)
        print(f"Detected encoding for {data_file}: {encoding}")

        # Load the dataset with detected encoding
        data = pd.read_csv(data_file, encoding=encoding)

        # Basic dataset information
        data_description = data.describe(include="all").to_string()
        missing_values = data.isnull().sum()

        # Select only numeric columns for correlation
        numeric_data = data.select_dtypes(include=["number"])
        correlation = numeric_data.corr() if not numeric_data.empty else None

        # Define output folder based on dataset name
        dataset_name = os.path.basename(data_file).split(".")[0]
        output_dir = os.path.join(".", dataset_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plots = []

        # 1. Correlation Heatmap
        if correlation is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
            heatmap_path = os.path.join(output_dir, f"correlation_heatmap_{dataset_name}.png")
            plt.title(f"Correlation Heatmap: {dataset_name}")
            plt.savefig(heatmap_path)
            plt.close()
            plots.append(heatmap_path)

        # 2. Missing Values Bar Chart
        if missing_values.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing_values.plot(kind="bar", color="orange")
            plt.title(f"Missing Values: {dataset_name}")
            missing_bar_path = os.path.join(output_dir, f"missing_values_{dataset_name}.png")
            plt.savefig(missing_bar_path)
            plt.close()
            plots.append(missing_bar_path)

        # Send the analysis to LLM
        summary = {
            "file_name": os.path.basename(data_file),
            "data_description": data_description,
            "missing_values": missing_values.to_string(),
            "plots": plots
        }

        # Use LLM to generate insights
        insights = get_llm_analysis(data_file, summary)

        # Create the README
        readme_content = f"""# Automated Data Analysis Report for {dataset_name}

### Dataset Overview
{data_description}

### Missing Values
{missing_values.to_string()}

### Visualizations
"""
        for plot in plots:
            readme_content += f"![{os.path.basename(plot)}]({plot})\n\n"

        readme_content += f"\n### Insights and Analysis\n{insights}\n"

        # Save the README file
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as readme_file:
            readme_file.write(readme_content)

        return output_dir

    except UnicodeDecodeError as e:
        print(f"Encoding error while processing {data_file}: {e}. Try converting the file to UTF-8.")
    except Exception as e:
        print(f"Error processing {data_file}: {e}")
        return None


# Function to get analysis from LLM
def get_llm_analysis(data_file, summary):
    try:
        # Construct the prompt string for the LLM
        prompt = f"""
        I have the following dataset for {os.path.basename(data_file)}:
        - Data Description: {summary["data_description"]}
        - Missing Values: {summary["missing_values"]}

        Based on the analysis above, generate insightful observations and any further analyses that could be beneficial for understanding this dataset.
        """

        # Make sure OpenAI API is initialized
        if not openai.api_key:
            raise ValueError("OpenAI API key is not set. Please set it in the environment variable AIPROXY_TOKEN.")

        # Request the analysis from OpenAI
        response = openai.Completion.create(
            engine="gpt-4o-mini",  # Ensure correct model is used
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )

        # Return the analysis text from the response
        return response.choices[0].text.strip()

    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        return "LLM analysis could not be completed due to an error."


# Main script
if __name__ == "__main__":
    # Check for file arguments
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py dataset1.csv dataset2.csv ...")
        sys.exit(1)

    # Iterate over provided dataset files
    for dataset_file in sys.argv[1:]:
        if os.path.exists(dataset_file):
            print(f"Analyzing {dataset_file}...")
            output_folder = analyze_dataset(dataset_file)
            if output_folder:
                print(f"Analysis complete! Check the folder '{output_folder}' for results.")
        else:
            print(f"Error: File {dataset_file} not found. Please check the path.")
