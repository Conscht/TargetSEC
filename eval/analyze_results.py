"""
Utility script to analyze and compare ECAPA-TDNN speaker similarity results.
Useful for examining results from multiple evaluation runs.
"""

import json
import sys
from pathlib import Path
from typing import Dict
import argparse


def load_results(json_path: str) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_results_summary(results: Dict, title: str = "Speaker Similarity Results"):
    """Print a formatted summary of results."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")
    
    # Per-class results
    if "per_class" in results:
        print("PER-CLASS INTRA-CLASS SIMILARITY (Speaker Consistency):")
        print("-" * 80)
        
        # Header
        print(f"{'Class':<8} {'Samples':<10} {'Pairs':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        
        # Data rows
        for class_id in sorted(results["per_class"].keys()):
            class_data = results["per_class"][class_id]
            print(f"{class_id:<8} {class_data['num_samples']:<10} {class_data['num_pairs']:<12} "
                  f"{class_data['mean_similarity']:<10.4f} {class_data['std_similarity']:<10.4f} "
                  f"{class_data['min_similarity']:<10.4f} {class_data['max_similarity']:<10.4f}")
        
        print()
    
    # Overall statistics
    if "overall" in results and results["overall"]:
        print("OVERALL STATISTICS:")
        print("-" * 80)
        overall = results["overall"]
        print(f"  Total pairs:          {overall.get('total_pairs', 'N/A')}")
        print(f"  Mean similarity:      {overall.get('mean_similarity', 'N/A'):.4f}")
        print(f"  Std similarity:       {overall.get('std_similarity', 'N/A'):.4f}")
        print(f"  Min similarity:       {overall.get('min_similarity', 'N/A'):.4f}")
        print(f"  Max similarity:       {overall.get('max_similarity', 'N/A'):.4f}")
        print(f"  Median similarity:    {overall.get('median_similarity', 'N/A'):.4f}")
        print()
    
    # Inter-class results
    if "inter_class" in results and results["inter_class"]:
        print("INTER-CLASS SIMILARITY:")
        print("-" * 80)
        
        # Header
        print(f"{'Class Pair':<15} {'Mean Sim':<12} {'Std':<12} {'Pairs':<8}")
        print("-" * 80)
        
        # Data rows
        for pair_key in sorted(results["inter_class"].keys()):
            pair_data = results["inter_class"][pair_key]
            print(f"{pair_key:<15} {pair_data['mean_similarity']:<12.4f} "
                  f"{pair_data['std_similarity']:<12.4f} {pair_data['num_pairs']:<8}")
        
        print()


def compare_results(results_list: list, labels: list = None):
    """Compare multiple evaluation results."""
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results_list))]
    
    print("\n" + "=" * 100)
    print("COMPARISON OF MULTIPLE EVALUATION RUNS")
    print("=" * 100 + "\n")
    
    # Compare per-class means
    print("PER-CLASS MEAN SIMILARITY COMPARISON:")
    print("-" * 100)
    
    # Header
    header = f"{'Class':<8}"
    for label in labels:
        header += f" {label:<15}"
    print(header)
    print("-" * 100)
    
    # Get all classes
    all_classes = set()
    for results in results_list:
        all_classes.update(results.get("per_class", {}).keys())
    
    # Data rows
    for class_id in sorted(all_classes):
        row = f"{class_id:<8}"
        for results in results_list:
            if class_id in results.get("per_class", {}):
                mean_sim = results["per_class"][class_id]["mean_similarity"]
                row += f" {mean_sim:<15.4f}"
            else:
                row += f" {'N/A':<15}"
        print(row)
    
    print("\n")
    
    # Compare overall means
    print("OVERALL MEAN SIMILARITY COMPARISON:")
    print("-" * 100)
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        if "overall" in results and results["overall"]:
            mean = results["overall"].get("mean_similarity", "N/A")
            std = results["overall"].get("std_similarity", "N/A")
            print(f"{label:<20} Mean: {mean:.4f}  |  Std: {std:.4f}")
        else:
            print(f"{label:<20} N/A")
    
    print()


def export_csv(results: Dict, output_path: str):
    """Export per-class results to CSV format."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Class', 'Samples', 'Pairs', 'Mean_Similarity', 'Std_Similarity',
            'Min_Similarity', 'Max_Similarity', 'Median_Similarity', 'Q25', 'Q75'
        ])
        
        # Data rows
        for class_id in sorted(results.get("per_class", {}).keys()):
            class_data = results["per_class"][class_id]
            writer.writerow([
                class_id,
                class_data['num_samples'],
                class_data['num_pairs'],
                f"{class_data['mean_similarity']:.6f}",
                f"{class_data['std_similarity']:.6f}",
                f"{class_data['min_similarity']:.6f}",
                f"{class_data['max_similarity']:.6f}",
                f"{class_data['median_similarity']:.6f}",
                f"{class_data['q25_similarity']:.6f}",
                f"{class_data['q75_similarity']:.6f}",
            ])
    
    print(f"Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ECAPA-TDNN speaker similarity evaluation results"
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        help="Path(s) to speaker_similarity_results.json file(s)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each results file (optional, for comparison)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple result files"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV file"
    )
    
    args = parser.parse_args()
    
    # Load results
    results_list = []
    for result_file in args.result_files:
        if not Path(result_file).exists():
            print(f"ERROR: File not found: {result_file}")
            sys.exit(1)
        results_list.append(load_results(result_file))
    
    # Set default labels
    if args.labels is None:
        args.labels = [Path(f).parent.name for f in args.result_files]
    
    # Display results
    if args.compare and len(results_list) > 1:
        compare_results(results_list, args.labels)
    else:
        # Print individual results
        for results, label in zip(results_list, args.labels):
            print_results_summary(results, title=f"Results: {label}")
    
    # Export to CSV if requested
    if args.export_csv and len(results_list) == 1:
        export_csv(results_list[0], args.export_csv)
    elif args.export_csv and len(results_list) > 1:
        print("Warning: CSV export only supported for single result file")


if __name__ == "__main__":
    main()
