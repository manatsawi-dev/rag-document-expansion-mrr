# MRR Comparison Report

## Overview
This report compares the Mean Reciprocal Rank (MRR) results for different text representation approaches:

1. Original text only
2. Original + Expanded text
3. Expanded text only

## Results

| Dataset                    |   Avg MRR@1 |   Avg MRR@3 |   Avg MRR@5 |   Avg MRR@10 |
|:---------------------------|------------:|------------:|------------:|-------------:|
| original_text              |        0.94 |      0.9533 |      0.9573 |       0.9573 |
| original_and_expanded_text |        0.96 |      0.9733 |      0.9733 |       0.9733 |
| only_expanded_text         |        0.9  |      0.9333 |      0.9333 |       0.9333 |

## Analysis

### Key Findings:

- Best performing approach: **original_and_expanded_text** with MRR@1 of 0.9600
