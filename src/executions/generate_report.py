from src.utils.report_utils import generate_mrr_comparison_report


def main():
    report_path = generate_mrr_comparison_report()
    print(f"Report generated successfully at: {report_path}")


if __name__ == "__main__":
    main()
