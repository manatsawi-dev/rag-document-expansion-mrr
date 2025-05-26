from src.executions.insert_documents import (
    insert_original_documents,
    insert_original_and_expanded_documents,
    insert_expanded_documents,
)

from src.executions.generate_question import generate_questions
from src.executions.evaluate_similarity_search import evaluate_similarity_search
from src.executions.evaluate_mrr import evaluate_mrr
from src.utils.report_utils import generate_mrr_comparison_report

import src.constants.constants as constants


def main():
    options = {
        "0": "Check your options",
        "1": "Insert original text into Qdrant",
        "2": "Insert original + expanded text into Qdrant",
        "3": "Insert only expanded text into Qdrant",
        "4": "Generate questions",
        "5": "Evaluate similarity search",
        "6": "Evaluate MRR",
        "7": "Generate MRR comparison report",
        "99": "Exit",
    }

    collections = [
        constants.ORIGINAL_TEXT_COLLECTION_NAME,
        constants.ORIGINAL_TEXT_AND_EXPANDED_COLLECTION_NAME,
        constants.ONLY_EXPANDED_COLLECTION_NAME,
    ]

    for key, value in options.items():
        print(f"{key}: {value}")

    while True:
        choice = input("Enter your choice (0: Check your options): ")

        if choice == "0":
            for key, value in options.items():
                print(f"{key}: {value}")
        elif choice == "1":
            print("Inserting original text into Qdrant...")
            # Call the function to insert original text
            insert_original_documents()
        elif choice == "2":
            print("Inserting original + expanded text into Qdrant...")
            # Call the function to insert original + expanded text
            insert_original_and_expanded_documents()
        elif choice == "3":
            print("Inserting only expanded text into Qdrant...")
            # Call the function to insert only expanded text
            insert_expanded_documents()
        elif choice == "4":
            print("Generating questions...")
            # Call the function to generate questions
            generate_questions()
        elif choice == "5":
            print("Evaluating similarity search...")
            # Call the function to evaluate similarity search

            evaluate_similarity_search(collection_name=collections[2])
        elif choice == "6":
            print("Evaluating MRR...")
            # Call the function to evaluate MRR
            evaluate_mrr(collection_name=collections[2])
        elif choice == "7":
            print("Generating MRR comparison report...")
            report_path = generate_mrr_comparison_report()
            print(f"Report generated successfully at: {report_path}")
        elif choice == "99":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
