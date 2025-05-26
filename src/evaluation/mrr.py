from typing import List, Dict, Set
import pandas as pd

from src.models.mrr_document import MRRDataset, MRRDatasetInfo

# ---------- CONFIGURABLE PARAMETERS ----------
TOP_Ks = [1, 3, 5, 10]  # You can change this to [1, 10, 20] as needed


class MRREvaluation:
    def __init__(self):
        pass

    # ---------- FUNCTION TO CALCULATE MRR@K ----------
    def reciprocal_rank(
        self, retrieved: List[str], relevant: Set[str], k: int
    ) -> float:
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    # ---------- MAIN FUNCTION FOR DATASET ----------
    def evaluate_dataset(
        self,
        dataset_information: MRRDatasetInfo,
        queries: List[MRRDataset],
        total_queries: int,
    ) -> tuple[pd.DataFrame, Dict]:
        results = []

        for q in queries:
            row = {"query_id": q.query_id}

            for k in TOP_Ks:
                row[f"MRR@{k}"] = self.reciprocal_rank(q.retrieved, q.relevant, k)

            results.append(row)

        df = pd.DataFrame(results)
        summary = {
            "Dataset": dataset_information.dataset_name,
            "Embedding": dataset_information.embedding_type,
            "Note": dataset_information.note,
            "Total Queries": total_queries,
        }
        for k in TOP_Ks:
            summary[f"Avg MRR@{k}"] = df[f"MRR@{k}"].mean()
            summary[f"Max MRR@{k}"] = df[f"MRR@{k}"].max()
            summary[f"Min MRR@{k}"] = df[f"MRR@{k}"].min()

        return df, summary
