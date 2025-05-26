import pandas as pd
import json
from pathlib import Path


def create_merged_summary_to_dataframes(summaries_list, dataset_names):
    dataframes = []

    for i, summary in enumerate(summaries_list):
        # Convert the dictionary to DataFrame if it's not already one
        if isinstance(summary, dict):
            df = pd.DataFrame([summary])  # Convert dict to single-row DataFrame
        else:
            df = summary

        # Add dataset name
        df["Dataset"] = dataset_names[i]
        dataframes.append(df)

    # Now concatenate the list of dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Reorder columns to put dataset name first
    cols = combined_df.columns.tolist()
    cols = ["Dataset"] + [col for col in cols if col != "Dataset"]
    return combined_df[cols]


def generate_mrr_comparison_report(
    output_path: str = "data/report/mrr_comparison_report.md",
):
    """
    Generate a markdown report comparing MRR results from different approaches.

    Args:
        output_path (str): Path where the markdown report will be saved
    """
    # Read the summary files
    summary_files = [
        "data/mrr_results/processed/original_text_summary.json",
        "data/mrr_results/processed/original_and_expanded_text_summary.json",
        "data/mrr_results/processed/only_expanded_text_summary.json",
    ]

    summaries = []
    for file_path in summary_files:
        with open(file_path, "r") as f:
            summaries.append(json.load(f))

    # Create DataFrame
    df = pd.DataFrame(summaries)

    # Select relevant columns
    mrr_columns = ["Dataset", "Avg MRR@1", "Avg MRR@3", "Avg MRR@5", "Avg MRR@10"]
    df_mrr = df[mrr_columns].copy()  # Create an explicit copy of the DataFrame

    # Format MRR values to 4 decimal places
    for col in mrr_columns[1:]:
        df_mrr[col] = df_mrr[col].apply(lambda x: f"{x:.4f}").astype(str)

    # Generate markdown content
    markdown_content = "# MRR Comparison Report\n\n"
    markdown_content += "## Overview\n"
    markdown_content += "This report compares the Mean Reciprocal Rank (MRR) results for different text representation approaches:\n\n"
    markdown_content += "1. Original text only\n"
    markdown_content += "2. Original + Expanded text\n"
    markdown_content += "3. Expanded text only\n\n"

    markdown_content += "## Results\n\n"
    markdown_content += df_mrr.to_markdown(index=False)

    markdown_content += "\n\n## Analysis\n\n"
    markdown_content += "### Key Findings:\n\n"

    # Get best performing approach
    best_mrr1 = df["Avg MRR@1"].max()
    best_approach = df.loc[df["Avg MRR@1"] == best_mrr1, "Dataset"].iloc[0]

    markdown_content += f"- Best performing approach: **{best_approach}** with MRR@1 of {best_mrr1:.4f}\n"
    markdown_content += "- Performance comparison:\n"

    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write the report
    with open(output_path, "w") as f:
        f.write(markdown_content)

    return output_path
