import os
import json

from src.utils.file_utils import read_jsonl_file, write_file
from src.models.mrr_document import MRRDatasetInfo, MRRDataset
from src.evaluation.mrr import MRREvaluation


DATA_DIR = "data/mrr_results/preprocessing"
SUMMARY_DIR = "data/mrr_results/processed"


def evaluate_mrr(
    collection_name: str,
):

    dataset_info = MRRDatasetInfo(
        dataset_name=collection_name,
        note="-",
        embedding_type=collection_name.replace("_", " "),
    )

    dataset: list[MRRDataset] = []

    files = [
        f
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f)) and f.endswith(".jsonl")
    ]
    # filter only file names that start with collection_name
    files = [f for f in files if f.startswith(collection_name)]

    if not files:
        print(f"No files found for collection name: {collection_name}")
        return
    else:
        file_name = files[0]
        data = read_jsonl_file(os.path.join(DATA_DIR, file_name))

        if len(data) == 0:
            print(f"No data found in file: {file_name}")
            return
        else:

            for item in data:
                dataset.append(
                    MRRDataset(
                        query_id=item["query_id"],
                        retrieved=item["retrieved"],
                        relevant=set(item["relevant"]),
                    )
                )

    mrr = MRREvaluation()

    _, summary = mrr.evaluate_dataset(
        dataset_information=dataset_info,
        queries=dataset,
        total_queries=len(dataset),
    )

    write_file(
        SUMMARY_DIR, f"{collection_name}_summary.json", json.dumps(summary, indent=4)
    )
