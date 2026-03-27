import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetQualityFilter:
    """
    Filters evaluated Luxembourgish dataset based on quality scores.
    Removes entries where any evaluation score is below the minimum acceptable threshold.
    
    Supports CSV, JSON, and JSONL formats for both input and output.
    """
    
    def __init__(self, min_acceptable_score: int = 2):
        """
        Initialize the filter with configuration.
        
        Args:
            min_acceptable_score: Minimum score to keep an entry (default: 2)
        """
        self.min_acceptable_score = min_acceptable_score
        self.score_columns = [
            'linguistic_quality',
            'factual_accuracy',
            'instruction_adherence',
            'helpfulness_relevance'
        ]
        self.content_columns = ['instruction', 'response', 'messages']
        self.statistics = {}
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV, JSON, or JSONL file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            DataFrame containing the dataset
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        logger.info(f"Loading dataset from: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.csv':
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV file with {len(df)} entries")
        elif suffix == '.json':
            df = pd.read_json(filepath)
            logger.info(f"Loaded JSON file with {len(df)} entries")
        elif suffix == '.jsonl':
            df = self._load_jsonl(filepath)
            logger.info(f"Loaded JSONL file with {len(df)} entries")
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use CSV, JSON, or JSONL.")
        
        # Validate required score columns
        missing_score_cols = [col for col in self.score_columns if col not in df.columns]
        if missing_score_cols:
            raise ValueError(f"Dataset missing required score columns: {missing_score_cols}")
        
        # Log available content columns
        available_content = [col for col in self.content_columns if col in df.columns]
        logger.info(f"Available content columns: {available_content}")
        
        return df
    
    def _load_jsonl(self, filepath: Path) -> pd.DataFrame:
        """
        Load dataset from JSONL (JSON Lines) file.
        
        Args:
            filepath: Path to the JSONL file
            
        Returns:
            DataFrame containing the dataset
        """
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        if not records:
            raise ValueError("No valid records found in JSONL file")
        
        return pd.DataFrame(records)
    
    def _save_jsonl(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save DataFrame to JSONL (JSON Lines) file.
        
        Args:
            df: DataFrame to save
            filepath: Path to output file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                record = row.to_dict()
                # Handle any nested structures (like messages column)
                json_line = json.dumps(record, ensure_ascii=False)
                f.write(json_line + '\n')
    
    def analyze_scores(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution of scores in the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing score statistics
        """
        stats = {
            'total_entries': len(df),
            'score_distributions': {},
            'entries_with_low_scores': {},
            'perfect_scores': 0,
            'score_combinations': {}
        }
        
        # Analyze each score column
        for col in self.score_columns:
            if col in df.columns:
                value_counts = df[col].value_counts().to_dict()
                stats['score_distributions'][col] = {
                    'score_1': value_counts.get(1, 0),
                    'score_2': value_counts.get(2, 0),
                    'score_3': value_counts.get(3, 0),
                    'mean': round(df[col].mean(), 3),
                    'median': df[col].median()
                }
                
                # Count entries with score 1 in this column
                stats['entries_with_low_scores'][col] = int((df[col] == 1).sum())
        
        # Count entries with all perfect scores (all 3s)
        perfect_mask = (df[self.score_columns] == 3).all(axis=1)
        stats['perfect_scores'] = int(perfect_mask.sum())
        
        # Analyze score combinations for entries with at least one low score
        low_score_df = df[(df[self.score_columns] == 1).any(axis=1)]
        if len(low_score_df) > 0:
            combo_counts = {}
            for _, row in low_score_df.iterrows():
                low_cols = tuple(sorted([col for col in self.score_columns if row[col] == 1]))
                combo_counts[low_cols] = combo_counts.get(low_cols, 0) + 1
            stats['score_combinations'] = {
                ', '.join(k): v for k, v in sorted(combo_counts.items(), key=lambda x: -x[1])
            }
        
        return stats
    
    def filter_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter dataset to remove entries with any low scores.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Tuple of (filtered_df, rejected_df)
        """
        logger.info("Filtering dataset based on quality scores...")
        
        # Create mask for entries to keep (no score below minimum)
        keep_mask = (df[self.score_columns] >= self.min_acceptable_score).all(axis=1)
        
        # Split into accepted and rejected
        filtered_df = df[keep_mask].copy()
        rejected_df = df[~keep_mask].copy()
        
        # Add rejection reason to rejected entries
        if len(rejected_df) > 0:
            rejection_reasons = []
            rejection_details = []
            for _, row in rejected_df.iterrows():
                low_scores = {col: int(row[col]) for col in self.score_columns 
                             if row[col] < self.min_acceptable_score}
                rejection_reasons.append(', '.join(low_scores.keys()))
                rejection_details.append(json.dumps(low_scores))
            rejected_df['rejection_reason'] = rejection_reasons
            rejected_df['rejection_scores'] = rejection_details
        
        logger.info(f"Filtering complete: {len(filtered_df)} accepted, {len(rejected_df)} rejected")
        
        return filtered_df, rejected_df
    
    def filter_by_composite_score(self, df: pd.DataFrame, 
                                   min_average: float = 2.0,
                                   weights: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Alternative filtering based on weighted average score.
        
        Args:
            df: DataFrame to filter
            min_average: Minimum weighted average score to keep an entry
            weights: Optional weights for each score column (default: equal weights)
            
        Returns:
            Tuple of (filtered_df, rejected_df)
        """
        if weights is None:
            weights = {col: 1.0 for col in self.score_columns}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted average
        df_copy = df.copy()
        df_copy['composite_score'] = sum(
            df_copy[col] * weight for col, weight in weights.items()
        )
        
        # Filter
        keep_mask = df_copy['composite_score'] >= min_average
        filtered_df = df_copy[keep_mask].copy()
        rejected_df = df_copy[~keep_mask].copy()
        
        return filtered_df, rejected_df
    
    def generate_report(self, original_stats: Dict, filtered_stats: Dict, 
                       rejected_count: int) -> str:
        """
        Generate a detailed filtering report.
        
        Args:
            original_stats: Statistics from original dataset
            filtered_stats: Statistics from filtered dataset
            rejected_count: Number of rejected entries
            
        Returns:
            Formatted report string
        """
        retention_rate = (filtered_stats['total_entries'] / original_stats['total_entries']) * 100
        
        report = f"""
{'='*70}
LUXEMBOURGISH DATASET QUALITY FILTERING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Minimum Acceptable Score: {self.min_acceptable_score}
{'='*70}

SUMMARY
-------
Original dataset size:  {original_stats['total_entries']:,} entries
Filtered dataset size:  {filtered_stats['total_entries']:,} entries
Rejected entries:       {rejected_count:,} entries
Retention rate:         {retention_rate:.2f}%

ORIGINAL DATASET SCORE DISTRIBUTION
------------------------------------"""
        
        for col in self.score_columns:
            if col in original_stats['score_distributions']:
                dist = original_stats['score_distributions'][col]
                total = original_stats['total_entries']
                report += f"""
{col}:
  Score 1 (Low):        {dist['score_1']:>6,} ({dist['score_1']/total*100:>5.1f}%)
  Score 2 (Acceptable): {dist['score_2']:>6,} ({dist['score_2']/total*100:>5.1f}%)
  Score 3 (Excellent):  {dist['score_3']:>6,} ({dist['score_3']/total*100:>5.1f}%)
  Mean: {dist['mean']:.3f}, Median: {dist['median']:.1f}"""
        
        report += f"""

Entries with all perfect scores: {original_stats['perfect_scores']:,} ({original_stats['perfect_scores']/original_stats['total_entries']*100:.2f}%)

FILTERED DATASET SCORE DISTRIBUTION
------------------------------------"""
        
        if filtered_stats['total_entries'] > 0:
            for col in self.score_columns:
                if col in filtered_stats['score_distributions']:
                    dist = filtered_stats['score_distributions'][col]
                    total = filtered_stats['total_entries']
                    report += f"""
{col}:
  Score 2 (Acceptable): {dist['score_2']:>6,} ({dist['score_2']/total*100:>5.1f}%)
  Score 3 (Excellent):  {dist['score_3']:>6,} ({dist['score_3']/total*100:>5.1f}%)
  Mean: {dist['mean']:.3f}, Median: {dist['median']:.1f}"""
            
            report += f"""

Entries with all perfect scores: {filtered_stats['perfect_scores']:,} ({filtered_stats['perfect_scores']/filtered_stats['total_entries']*100:.2f}%)"""
        else:
            report += "\nNo entries remaining after filtering."
        
        report += f"""

REJECTION BREAKDOWN BY SCORE COLUMN
------------------------------------"""
        
        for col in self.score_columns:
            if col in original_stats['entries_with_low_scores']:
                count = original_stats['entries_with_low_scores'][col]
                pct = count / original_stats['total_entries'] * 100
                report += f"""
{col}: {count:,} entries had score 1 ({pct:.2f}%)"""
        
        # Add score combination analysis
        if original_stats.get('score_combinations'):
            report += f"""

REJECTION PATTERNS (Score Column Combinations)
----------------------------------------------"""
            for combo, count in list(original_stats['score_combinations'].items())[:10]:
                pct = count / rejected_count * 100
                report += f"""
{combo}: {count:,} entries ({pct:.1f}% of rejected)"""
        
        report += f"""

{'='*70}
"""
        return report
    
    def save_dataset(self, df: pd.DataFrame, base_path: str, 
                    suffix: str = "", output_format: str = "all") -> Dict[str, str]:
        """
        Save dataset in specified format(s).
        
        Args:
            df: DataFrame to save
            base_path: Base path for output files (without extension)
            suffix: Optional suffix to add to filename
            output_format: Output format - "json", "csv", "jsonl", or "all"
            
        Returns:
            Dictionary of format -> filepath
        """
        base_path = Path(base_path).with_suffix('')
        
        # Create filenames with suffix
        if suffix:
            base_name = f"{base_path}_{suffix}"
        else:
            base_name = str(base_path)
        
        output_paths = {}
        
        formats_to_save = ['json', 'csv', 'jsonl'] if output_format == "all" else [output_format]
        
        for fmt in formats_to_save:
            filepath = f"{base_name}.{fmt}"
            
            if fmt == 'json':
                logger.info(f"Saving JSON to: {filepath}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
                output_paths['json'] = filepath
                
            elif fmt == 'csv':
                logger.info(f"Saving CSV to: {filepath}")
                df.to_csv(filepath, index=False, encoding='utf-8')
                output_paths['csv'] = filepath
                
            elif fmt == 'jsonl':
                logger.info(f"Saving JSONL to: {filepath}")
                self._save_jsonl(df, filepath)
                output_paths['jsonl'] = filepath
        
        return output_paths
    
    def prepare_for_training(self, df: pd.DataFrame, 
                             include_scores: bool = False,
                             output_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare filtered dataset for training by selecting relevant columns.
        
        Args:
            df: DataFrame to prepare
            include_scores: Whether to include score columns in output
            output_columns: Specific columns to include (default: instruction, response, messages)
            
        Returns:
            DataFrame with only training-relevant columns
        """
        if output_columns is None:
            output_columns = [col for col in self.content_columns if col in df.columns]
        
        if include_scores:
            output_columns = output_columns + [col for col in self.score_columns if col in df.columns]
        
        available_cols = [col for col in output_columns if col in df.columns]
        
        if not available_cols:
            raise ValueError(f"None of the specified columns found in dataset: {output_columns}")
        
        logger.info(f"Preparing dataset with columns: {available_cols}")
        return df[available_cols].copy()


def filter_evaluated_dataset(
    input_path: str,
    output_base_path: str = "filtered_dataset",
    output_format: str = "jsonl",
    save_rejected: bool = True,
    generate_report_file: bool = True,
    min_acceptable_score: int = 2,
    prepare_training_set: bool = True,
    include_scores_in_training: bool = False
) -> Dict[str, Any]:
    """
    Main function to filter an evaluated Luxembourgish dataset.
    
    Args:
        input_path: Path to input dataset (CSV, JSON, or JSONL)
        output_base_path: Base path for output files (without extension)
        output_format: Output format - "json", "csv", "jsonl", or "all"
        save_rejected: Whether to save rejected entries to separate files
        generate_report_file: Whether to generate a text report file
        min_acceptable_score: Minimum score to keep an entry (default: 2)
        prepare_training_set: Whether to create a training-ready version (without scores)
        include_scores_in_training: Whether to include scores in training set
        
    Returns:
        Dictionary containing filtering results and statistics
    """
    # Initialize filter
    filter_obj = DatasetQualityFilter(min_acceptable_score=min_acceptable_score)
    
    # Load dataset
    df = filter_obj.load_dataset(input_path)
    logger.info(f"Successfully loaded dataset with {len(df)} entries")
    
    # Log column information
    logger.info(f"Dataset columns: {list(df.columns)}")
    
    # Analyze original dataset
    logger.info("Analyzing original dataset scores...")
    original_stats = filter_obj.analyze_scores(df)
    
    # Filter dataset
    filtered_df, rejected_df = filter_obj.filter_dataset(df)
    
    # Analyze filtered dataset
    logger.info("Analyzing filtered dataset scores...")
    filtered_stats = filter_obj.analyze_scores(filtered_df)
    
    # Generate report
    report = filter_obj.generate_report(original_stats, filtered_stats, len(rejected_df))
    print(report)
    
    # Prepare output paths dictionary
    output_files = {}
    
    # Save filtered dataset (full version with scores)
    filtered_paths = filter_obj.save_dataset(
        filtered_df, output_base_path, "filtered", output_format
    )
    output_files['filtered'] = filtered_paths
    logger.info(f"Filtered dataset saved: {len(filtered_df)} entries")
    
    # Save training-ready version if requested
    if prepare_training_set:
        training_df = filter_obj.prepare_for_training(
            filtered_df, 
            include_scores=include_scores_in_training
        )
        training_paths = filter_obj.save_dataset(
            training_df, output_base_path, "training", output_format
        )
        output_files['training'] = training_paths
        logger.info(f"Training dataset saved: {len(training_df)} entries")
    
    # Save rejected entries if requested
    if save_rejected and len(rejected_df) > 0:
        rejected_paths = filter_obj.save_dataset(
            rejected_df, output_base_path, "rejected", output_format
        )
        output_files['rejected'] = rejected_paths
        logger.info(f"Rejected entries saved: {len(rejected_df)} entries")
    
    # Save report to file if requested
    if generate_report_file:
        report_path = f"{output_base_path}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        output_files['report'] = report_path
        logger.info(f"Report saved to: {report_path}")
    
    # Prepare results
    results = {
        'original_count': len(df),
        'filtered_count': len(filtered_df),
        'rejected_count': len(rejected_df),
        'retention_rate': round((len(filtered_df) / len(df)) * 100, 2),
        'output_files': output_files,
        'statistics': {
            'original': original_stats,
            'filtered': filtered_stats
        }
    }
    
    logger.info(f"""
    ==========================================
    FILTERING COMPLETE
    ==========================================
    Original entries:  {results['original_count']:,}
    Accepted entries:  {results['filtered_count']:,}
    Rejected entries:  {results['rejected_count']:,}
    Retention rate:    {results['retention_rate']:.2f}%
    ==========================================
    """)
    
    return results


# Example usage
if __name__ == "__main__":
    # Configuration
    INPUT_PATH = "results/evaluated_merged.jsonl"  # Now supports .jsonl
    OUTPUT_BASE = "data/LuxIT_large"
    
    # Run filtering
    results = filter_evaluated_dataset(
        input_path=INPUT_PATH,
        output_base_path=OUTPUT_BASE,
        output_format="jsonl",  # Primary output format (can also be "json", "csv", or "all")
        save_rejected=True,  # Save rejected entries for review
        generate_report_file=True,  # Generate detailed report
        min_acceptable_score=2,  # Reject entries with any score of 1
        prepare_training_set=True,  # Create training-ready version
        include_scores_in_training=False  # Don't include scores in training set
    )
    
    print(f"\nFiltering complete!")
    print(f"Output files: {json.dumps(results['output_files'], indent=2)}")
    print(f"Retention rate: {results['retention_rate']:.2f}%")