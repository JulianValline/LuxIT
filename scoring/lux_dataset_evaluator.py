#!/usr/bin/env python3
"""
Luxembourgish Dataset Evaluator with Parallel Processing Support

This script evaluates Luxembourgish instruction-response pairs using OpenAI's API.
Supports parallel execution via dataset splitting and worker IDs.

Usage:
    # Single worker (all data):
    python evaluate_luxembourgish_parallel.py --dataset data.jsonl --output results
    
    # Parallel execution (run each in separate tmux window):
    python evaluate_luxembourgish_parallel.py --dataset data.jsonl --output results --num-workers 4 --worker-id 0
    python evaluate_luxembourgish_parallel.py --dataset data.jsonl --output results --num-workers 4 --worker-id 1
    python evaluate_luxembourgish_parallel.py --dataset data.jsonl --output results --num-workers 4 --worker-id 2
    python evaluate_luxembourgish_parallel.py --dataset data.jsonl --output results --num-workers 4 --worker-id 3
    
    # Merge results after all workers complete:
    python evaluate_luxembourgish_parallel.py --merge --output results --num-workers 4
"""

import json
import time
import os
import random
import argparse
import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from openai import OpenAI

# Load environment variables
load_dotenv()


def setup_logging(worker_id: Optional[int] = None, log_dir: str = "logs") -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    
    Args:
        worker_id: Worker ID for parallel processing (None for single worker)
        log_dir: Directory for log files
    
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"evaluator_worker_{worker_id}" if worker_id is not None else "evaluator")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler - detailed logging
    worker_suffix = f"_worker{worker_id}" if worker_id is not None else ""
    log_file = Path(log_dir) / f"evaluation{worker_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - Worker{worker_id} - %(levelname)s - %(message)s'.format(
            worker_id=worker_id if worker_id is not None else "Main"
        ),
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_dataset(dataset_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load dataset from CSV, JSON, or JSONL format.
    
    Args:
        dataset_path: Path to the dataset file
        logger: Logger instance
    
    Returns:
        DataFrame with the dataset
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.json'):
        df = pd.read_json(dataset_path)
    elif dataset_path.endswith('.jsonl'):
        # Load JSONL file line by line
        records = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} records from JSONL file")
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}. Use .csv, .json, or .jsonl")
    
    # Validate required columns
    required_columns = ['instruction', 'response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    logger.info(f"Dataset loaded: {len(df)} entries, columns: {list(df.columns)}")
    return df


def save_dataset(df: pd.DataFrame, output_path: str, logger: logging.Logger):
    """
    Save dataset to CSV, JSON, or JSONL format based on extension.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        logger: Logger instance
    """
    logger.info(f"Saving dataset to: {output_path}")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format from the actual extension (handle .tmp.jsonl case)
    path_lower = output_path.lower()
    if path_lower.endswith('.csv'):
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif path_lower.endswith('.json'):
        df.to_json(output_path, orient='records', force_ascii=False, indent=2)
    elif path_lower.endswith('.jsonl'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Convert row to dict and handle any non-serializable types
                row_dict = {}
                for key, value in row.to_dict().items():
                    # Handle lists/arrays (like 'messages' column) - keep as-is
                    if isinstance(value, (list, dict)):
                        row_dict[key] = value
                    # Handle numpy arrays
                    elif hasattr(value, 'tolist'):
                        row_dict[key] = value.tolist()
                    # Handle NA/None values (must check after list/array check)
                    elif value is None:
                        row_dict[key] = None
                    elif isinstance(value, float) and pd.isna(value):
                        row_dict[key] = None
                    # Handle basic types
                    elif isinstance(value, (int, float, str, bool)):
                        row_dict[key] = value
                    # Fallback: convert to string
                    else:
                        try:
                            # Try to check if it's a scalar NA
                            if pd.isna(value):
                                row_dict[key] = None
                            else:
                                row_dict[key] = str(value)
                        except (ValueError, TypeError):
                            # If pd.isna fails (e.g., for arrays), keep the value
                            row_dict[key] = value
                f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported output format: {output_path}. Expected .csv, .json, or .jsonl")
    
    logger.info(f"Saved {len(df)} entries to {output_path}")


class LuxembourgishDatasetEvaluator:
    """
    Evaluator for Luxembourgish instruction-response pairs using OpenAI's API.
    
    Key Features:
    - Checkpoint-based recovery for interrupted processing
    - Parallel processing support with worker IDs
    - Incremental evaluation support
    - Robust retry logic with exponential backoff
    - Preserves all original dataset columns (instruction, response, source_type, messages)
    """
    
    SCORE_COLUMNS = ['linguistic_quality', 'factual_accuracy', 
                     'instruction_adherence', 'helpfulness_relevance']
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-5-mini",
        checkpoint_dir: str = "checkpoints",
        worker_id: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (e.g., "gpt-5-mini", "gpt-5")
            checkpoint_dir: Directory for checkpoint files
            worker_id: Worker ID for parallel processing
            logger: Logger instance
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.worker_id = worker_id
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup checkpoint file path
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        worker_suffix = f"_worker{worker_id}" if worker_id is not None else ""
        self.checkpoint_file = Path(checkpoint_dir) / f"checkpoint{worker_suffix}.json"
        
        # Load checkpoint
        self.processed_ids, self.checkpoint_scores = self._load_checkpoint()
        self.logger.info(f"Initialized evaluator (model: {model}, worker: {worker_id})")
        self.logger.info(f"Checkpoint file: {self.checkpoint_file}")
        self.logger.info(f"Previously processed entries: {len(self.processed_ids)}")
    
    def _load_checkpoint(self) -> tuple:
        """Load previously processed IDs and scores from checkpoint file.
        
        Returns:
            Tuple of (processed_ids set, scores dict)
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    processed = set(checkpoint_data.get('processed_ids', []))
                    scores = checkpoint_data.get('scores', {})
                    self.logger.info(f"Loaded checkpoint: {len(processed)} IDs, {len(scores)} scores")
                    return processed, scores
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
                return set(), {}
        return set(), {}
    
    def _save_checkpoint(self):
        """Save processed IDs and scores to checkpoint file.
        
        This APPENDS to existing data (old entries are loaded at startup
        and kept in memory, so saving writes old + new entries).
        """
        checkpoint_data = {
            'processed_ids': sorted(list(self.processed_ids)),
            'scores': self.checkpoint_scores,
            'worker_id': self.worker_id,
            'count': len(self.processed_ids),
            'last_updated': datetime.now().isoformat()
        }
        
        # Write to temp file first, then rename (atomic operation)
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f)
            temp_file.rename(self.checkpoint_file)
            self.logger.debug(f"Checkpoint saved: {len(self.processed_ids)} entries")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            raise
    
    def _generate_entry_id(self, instruction: str, response: str) -> str:
        """Generate a stable ID based on content hash."""
        content = f"{instruction}||{response}"
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return f"EVAL_{hash_obj.hexdigest()[:16]}"
    
    def _create_evaluation_prompt(self, instruction: str, response: str) -> str:
        """Create the evaluation prompt for the LLM judge."""
        return f"""You are an expert evaluator of Luxembourgish text quality. Your task is to evaluate the following instruction-response pair written in Luxembourgish based on four specific criteria.

IMPORTANT: The texts below are in Luxembourgish. You must evaluate them as Luxembourgish texts, NOT as German, French, or any other language. Luxembourgish has its own distinct grammar, vocabulary, and spelling conventions.

INSTRUCTION (in Luxembourgish):
{instruction}

RESPONSE (in Luxembourgish):
{response}

EVALUATION CRITERIA:

1. linguistic_quality (Linguistic Quality):
- Score 1 (Poor): Contains significant grammatical errors, spelling mistakes, or unnatural phrasing in Luxembourgish. Text that is actually German or French instead of proper Luxembourgish should receive this score.
- Score 2 (Acceptable): Mostly correct Luxembourgish, but has minor errors or sounds slightly robotic/unnatural. May mix in too many loan words unnecessarily.
- Score 3 (Excellent): Fluent, idiomatic, and grammatically perfect Luxembourgish. Natural-sounding text that a native speaker would produce.

2. factual_accuracy (Factual Accuracy):
- Score 1 (Incorrect): Contains factual errors that contradict the source text or general knowledge.
- Score 2 (Mostly Correct): Mostly accurate but might have minor inaccuracies or omissions.
- Score 3 (Perfect): Completely accurate according to the source text and factual knowledge.

3. instruction_adherence (Instruction Following):
- Score 1 (Not Followed): Fails to follow the core instruction (e.g., provides a summary when asked for a list).
- Score 2 (Partially Followed): Follows the main instruction but misses a constraint (e.g., writes 4 bullet points when asked for 3, wrong format, or incorrect tone).
- Score 3 (Fully Followed): Perfectly follows all parts of the instruction, including constraints like length, format, and tone.

4. helpfulness_relevance (Helpfulness and Relevance):
- Score 1 (Not Helpful): The instruction is nonsensical, irrelevant to any reasonable context, or the response is unhelpful/off-topic.
- Score 2 (Somewhat Helpful): The instruction is plausible but not very insightful or creative. The response addresses the instruction but in a basic way.
- Score 3 (Very Helpful): A genuinely useful, interesting, or creative instruction that elicits a helpful, comprehensive response.

CRITICAL INSTRUCTIONS:
- Respond ONLY with a JSON object containing the four scores.
- Each score must be an integer: 1, 2, or 3.
- Do NOT include any explanations, comments, or additional text outside the JSON.
- Evaluate the text AS LUXEMBOURGISH, not as any other language.

JSON FORMAT:
{{
    "linguistic_quality": <score 1-3>,
    "factual_accuracy": <score 1-3>,
    "instruction_adherence": <score 1-3>,
    "helpfulness_relevance": <score 1-3>
}}"""
    
    def _call_api(
        self, 
        instruction: str, 
        response: str,
        max_retries: int = 10,
        base_delay: float = 2.0,
        max_delay: float = 120.0
    ) -> Optional[Dict[str, int]]:
        """
        Call OpenAI API with robust retry logic.
        
        Args:
            instruction: The instruction text
            response: The response text
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
        
        Returns:
            Dictionary with scores or None if failed
        """
        prompt = self._create_evaluation_prompt(instruction, response)
        
        for attempt in range(max_retries):
            try:
                api_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert evaluator of Luxembourgish text quality. You understand the distinct characteristics of Luxembourgish as separate from German and French. Always respond with valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1, #0.1,  # Low temperature for consistent evaluation (only for GPT-5)
                    response_format={"type": "json_object"}
                )
                
                # Parse the JSON response
                scores = json.loads(api_response.choices[0].message.content)
                
                # Validate response
                if all(key in scores for key in self.SCORE_COLUMNS):
                    for key in self.SCORE_COLUMNS:
                        if scores[key] not in [1, 2, 3]:
                            raise ValueError(f"Invalid score for {key}: {scores[key]}")
                    return scores
                else:
                    raise ValueError(f"Missing keys in response: {scores}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Rate limit handling
                if 'rate limit' in error_msg or '429' in str(e) or 'rate_limit' in error_msg:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay = delay * (0.5 + random.random())  # Add jitter
                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    
                # Network issues
                elif 'timeout' in error_msg or 'connection' in error_msg:
                    delay = min(base_delay * (1.5 ** attempt), 60)
                    self.logger.warning(
                        f"Network issue (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    
                # Other errors
                else:
                    self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        self.logger.error(f"Failed after {max_retries} attempts: {e}")
                        return None
        
        return None
    
    def evaluate_dataset(
        self,
        dataset: pd.DataFrame,
        output_path: str,
        checkpoint_interval: int = 10,
        save_interval: int = 100,
        rate_limit_delay: float = 0.5,
        force_reevaluate: bool = False
    ) -> pd.DataFrame:
        """
        Evaluate the dataset with checkpoint recovery.
        
        Args:
            dataset: DataFrame with 'instruction' and 'response' columns
            output_path: Path to save results
            checkpoint_interval: Save checkpoint every N entries
            save_interval: Save results to file every N entries
            rate_limit_delay: Delay between API calls
            force_reevaluate: Re-evaluate even if already processed
        
        Returns:
            Evaluated DataFrame
        """
        result_df = dataset.copy()
        
        # Initialize score columns
        for col in self.SCORE_COLUMNS:
            if col not in result_df.columns:
                result_df[col] = None
        
        # Generate entry IDs
        result_df['__eval_id__'] = result_df.apply(
            lambda row: self._generate_entry_id(row['instruction'], row['response']),
            axis=1
        )
        
        # Handle duplicates
        id_counts = result_df['__eval_id__'].value_counts()
        duplicates = id_counts[id_counts > 1]
        if len(duplicates) > 0:
            self.logger.warning(f"Found {len(duplicates)} duplicate instruction-response pairs")
            # Add suffix to duplicates
            seen = {}
            new_ids = []
            for idx, eval_id in enumerate(result_df['__eval_id__']):
                if eval_id in seen:
                    seen[eval_id] += 1
                    new_ids.append(f"{eval_id}_DUP{seen[eval_id]:03d}")
                else:
                    seen[eval_id] = 0
                    new_ids.append(eval_id)
            result_df['__eval_id__'] = new_ids
        
        # Try to load previous results
        if os.path.exists(output_path) and not force_reevaluate:
            try:
                previous_df = load_dataset(output_path, self.logger)
                self.logger.info(f"Found previous results: {len(previous_df)} entries")
                
                # Merge previous scores
                if '__eval_id__' in previous_df.columns:
                    prev_scores = previous_df.set_index('__eval_id__')[self.SCORE_COLUMNS]
                    for idx, row in result_df.iterrows():
                        eval_id = row['__eval_id__']
                        if eval_id in prev_scores.index:
                            prev_row = prev_scores.loc[eval_id]
                            if prev_row.notna().all():
                                for col in self.SCORE_COLUMNS:
                                    result_df.at[idx, col] = prev_row[col]
                                self.processed_ids.add(eval_id)
                    
                    self.logger.info(f"Merged {len(self.processed_ids)} previously evaluated entries")
            except Exception as e:
                self.logger.warning(f"Could not load previous results: {e}")
        
        # Restore scores from checkpoint
        if self.checkpoint_scores and not force_reevaluate:
            restored_count = 0
            for idx, row in result_df.iterrows():
                eval_id = row['__eval_id__']
                if eval_id in self.checkpoint_scores:
                    scores = self.checkpoint_scores[eval_id]
                    for col in self.SCORE_COLUMNS:
                        if col in scores:
                            result_df.at[idx, col] = scores[col]
                    self.processed_ids.add(eval_id)
                    restored_count += 1
            if restored_count > 0:
                self.logger.info(f"Restored {restored_count} entries from checkpoint scores")
        
        # Statistics
        total = len(result_df)
        processed = 0
        skipped = 0
        failed = 0
        failed_ids = []
        
        start_time = time.time()
        
        self.logger.info(f"Starting evaluation of {total} entries...")
        self.logger.info(f"Already processed: {len(self.processed_ids)} entries")
        
        try:
            for idx, row in result_df.iterrows():
                eval_id = row['__eval_id__']
                
                # Skip if already processed
                if eval_id in self.processed_ids and not force_reevaluate:
                    skipped += 1
                    continue
                
                # Skip if already has scores
                if not force_reevaluate:
                    row_scores = [row.get(col) for col in self.SCORE_COLUMNS]
                    if all(pd.notna(s) for s in row_scores):
                        self.processed_ids.add(eval_id)
                        skipped += 1
                        continue
                
                # Evaluate
                scores = self._call_api(row['instruction'], row['response'])
                
                if scores:
                    for col, value in scores.items():
                        result_df.at[idx, col] = value
                    self.processed_ids.add(eval_id)
                    # Store scores in checkpoint for recovery
                    self.checkpoint_scores[eval_id] = scores
                    processed += 1
                    
                    # Progress logging
                    if processed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = (total - processed - skipped) / rate if rate > 0 else 0
                        self.logger.info(
                            f"Progress: {processed + skipped}/{total} "
                            f"({(processed + skipped) / total * 100:.1f}%) | "
                            f"Rate: {rate:.2f}/s | "
                            f"ETA: {remaining / 60:.1f}min"
                        )
                    
                    # Save checkpoint (every checkpoint_interval entries)
                    if processed % checkpoint_interval == 0:
                        self._save_checkpoint()
                    
                    # Save results periodically
                    if processed % save_interval == 0:
                        self._save_results(result_df, output_path)
                        self.logger.info(f"Results saved ({processed} new entries processed)")
                else:
                    failed += 1
                    failed_ids.append(eval_id)
                    self.logger.error(f"Failed to evaluate entry: {eval_id[:20]}...")
                
                # Rate limiting
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay * (0.8 + 0.4 * random.random()))
                    
        except KeyboardInterrupt:
            self.logger.warning("Keyboard interrupt received! Saving progress...")
            self._save_checkpoint()
            self._save_results(result_df, output_path)
            self.logger.info(f"Progress saved. Processed {processed} entries before interrupt.")
            self.logger.info(f"Restart the script with the same arguments to resume.")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.logger.warning("Attempting to save progress before exit...")
            try:
                self._save_checkpoint()
                self._save_results(result_df, output_path)
                self.logger.info(f"Progress saved. Processed {processed} entries before crash.")
                self.logger.info(f"Restart the script with the same arguments to resume.")
            except Exception as save_error:
                self.logger.error(f"Failed to save progress: {save_error}")
            raise
        
        # Final save
        self._save_checkpoint()
        self._save_results(result_df, output_path)
        
        # Summary
        elapsed = time.time() - start_time
        self.logger.info(f"""
========================================
Evaluation Complete (Worker {self.worker_id})
========================================
Total entries:     {total}
Newly processed:   {processed}
Skipped:           {skipped}
Failed:            {failed}
Time elapsed:      {elapsed / 60:.1f} minutes
Average rate:      {processed / elapsed if elapsed > 0 else 0:.2f} entries/sec
Output saved to:   {output_path}
========================================
        """)
        
        if failed_ids:
            self.logger.warning(f"Failed entry IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
        
        # Clean up temp column
        result_df = result_df.drop(columns=['__eval_id__'])
        
        return result_df
    
    def _save_results(self, df: pd.DataFrame, output_path: str):
        """Save results with atomic write."""
        save_df = df.copy()
        
        # Create temp path with correct extension (e.g., file.jsonl -> file.tmp.jsonl)
        path = Path(output_path)
        temp_path = str(path.parent / f"{path.stem}.tmp{path.suffix}")
        backup_path = str(path.parent / f"{path.stem}.bak{path.suffix}")
        
        try:
            # Ensure output directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to temp file first
            save_dataset(save_df, temp_path, self.logger)
            
            # Atomic rename with backup
            if os.path.exists(output_path):
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(output_path, backup_path)
            os.rename(temp_path, output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise


def split_dataset_for_workers(
    dataset_path: str, 
    num_workers: int, 
    output_dir: str = "worker_splits",
    logger: logging.Logger = None
) -> List[str]:
    """
    Split dataset into chunks for parallel processing.
    
    Args:
        dataset_path: Path to the full dataset
        num_workers: Number of workers/chunks
        output_dir: Directory to save split files
        logger: Logger instance
    
    Returns:
        List of paths to split files
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Splitting dataset into {num_workers} chunks...")
    
    # Load full dataset
    df = load_dataset(dataset_path, logger)
    total = len(df)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Split into chunks
    chunk_size = total // num_workers
    split_paths = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_workers - 1 else total
        
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        # Determine output format (same as input)
        ext = Path(dataset_path).suffix
        split_path = Path(output_dir) / f"split_worker{i}{ext}"
        
        save_dataset(chunk_df, str(split_path), logger)
        split_paths.append(str(split_path))
        
        logger.info(f"Worker {i}: {len(chunk_df)} entries ({start_idx} to {end_idx-1})")
    
    logger.info(f"Dataset split complete. Files saved to: {output_dir}/")
    return split_paths


def merge_worker_results(
    output_base: str,
    num_workers: int,
    output_format: str = "jsonl",
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Merge results from all workers into a single file.
    
    Args:
        output_base: Base path for output files (without extension)
        num_workers: Number of workers
        output_format: Output format (csv, json, jsonl)
        logger: Logger instance
    
    Returns:
        Merged DataFrame
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Merging results from {num_workers} workers...")
    
    all_dfs = []
    
    for i in range(num_workers):
        # Try different extensions
        for ext in ['.jsonl', '.csv', '.json']:
            worker_path = f"{output_base}_worker{i}{ext}"
            if os.path.exists(worker_path):
                df = load_dataset(worker_path, logger)
                df['__worker_id__'] = i
                all_dfs.append(df)
                logger.info(f"Loaded worker {i}: {len(df)} entries from {worker_path}")
                break
        else:
            logger.warning(f"No results found for worker {i}")
    
    if not all_dfs:
        raise ValueError("No worker results found!")
    
    # Merge all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove worker ID column
    merged_df = merged_df.drop(columns=['__worker_id__'])
    
    # Remove duplicate eval IDs (keep first occurrence)
    if '__eval_id__' in merged_df.columns:
        initial_len = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['__eval_id__'], keep='first')
        merged_df = merged_df.drop(columns=['__eval_id__'])
        if len(merged_df) < initial_len:
            logger.info(f"Removed {initial_len - len(merged_df)} duplicate entries")
    
    # Save merged results
    merged_path = f"{output_base}_merged.{output_format}"
    save_dataset(merged_df, merged_path, logger)
    
    # Calculate statistics
    score_columns = LuxembourgishDatasetEvaluator.SCORE_COLUMNS
    evaluated = merged_df[score_columns].notna().all(axis=1).sum()
    
    logger.info(f"""
========================================
Merge Complete
========================================
Total entries:     {len(merged_df)}
Fully evaluated:   {evaluated}
Output saved to:   {merged_path}
========================================
    """)
    
    # Print score distribution
    if evaluated > 0:
        logger.info("Score distributions:")
        for col in score_columns:
            dist = merged_df[col].value_counts().sort_index()
            logger.info(f"  {col}: {dict(dist)}")
    
    return merged_df


def generate_tmux_commands(
    dataset_path: str,
    output_base: str,
    num_workers: int,
    api_key_env: str = "OPENAI_API_KEY",
    model: str = "gpt-5-mini",
    rate_limit_delay: float = 0.5
) -> str:
    """
    Generate tmux commands for parallel execution.
    
    Args:
        dataset_path: Path to dataset
        output_base: Base output path
        num_workers: Number of parallel workers
        api_key_env: Environment variable name for API key
        model: Model to use
        rate_limit_delay: Delay between API calls
    
    Returns:
        String with tmux commands
    """
    script_name = "evaluate_luxembourgish_parallel.py"
    
    commands = f"""
# =====================================================
# Parallel Evaluation Setup - {num_workers} Workers
# =====================================================

# 1. First, create a new tmux session:
tmux new-session -d -s eval

# 2. Create {num_workers} windows and start workers:
"""
    
    for i in range(num_workers):
        worker_output = f"{output_base}_worker{i}.jsonl"
        cmd = (
            f"python {script_name} "
            f"--dataset {dataset_path} "
            f"--output {worker_output} "
            f"--num-workers {num_workers} "
            f"--worker-id {i} "
            f"--model {model} "
            f"--rate-limit-delay {rate_limit_delay}"
        )
        
        if i == 0:
            commands += f"""
# Window {i} (main window):
tmux send-keys -t eval '{cmd}' Enter
"""
        else:
            commands += f"""
# Window {i}:
tmux new-window -t eval
tmux send-keys -t eval '{cmd}' Enter
"""
    
    commands += f"""
# 3. Attach to the tmux session to monitor:
tmux attach-session -t eval

# 4. Switch between windows:
#    Ctrl+B, then window number (0-{num_workers-1})
#    Or: Ctrl+B, n (next) / Ctrl+B, p (previous)

# 5. After ALL workers complete, merge results:
python {script_name} --merge --output {output_base} --num-workers {num_workers}

# =====================================================
# Alternative: Run directly in separate terminals
# =====================================================
"""
    
    for i in range(num_workers):
        worker_output = f"{output_base}_worker{i}.jsonl"
        commands += f"""
# Terminal {i+1}:
python {script_name} --dataset {dataset_path} --output {worker_output} --num-workers {num_workers} --worker-id {i} --model {model}
"""
    
    return commands


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate Luxembourgish instruction-response pairs using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single worker (evaluate all):
  python %(prog)s --dataset data.jsonl --output results.jsonl
  
  # Parallel execution (4 workers):
  python %(prog)s --dataset data.jsonl --output results --num-workers 4 --worker-id 0
  python %(prog)s --dataset data.jsonl --output results --num-workers 4 --worker-id 1
  # ... etc
  
  # Merge results:
  python %(prog)s --merge --output results --num-workers 4
  
  # Generate tmux commands:
  python %(prog)s --generate-tmux --dataset data.jsonl --output results --num-workers 4
        """
    )
    
    # Mode selection
    parser.add_argument('--merge', action='store_true',
                        help='Merge results from multiple workers')
    parser.add_argument('--generate-tmux', action='store_true',
                        help='Generate tmux commands for parallel execution')
    parser.add_argument('--split-only', action='store_true',
                        help='Only split the dataset, do not evaluate')
    
    # Data paths
    parser.add_argument('--dataset', type=str,
                        help='Path to input dataset (CSV, JSON, or JSONL)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path/base (extension added automatically for workers)')
    
    # Parallel processing
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--worker-id', type=int, default=None,
                        help='Worker ID (0 to num-workers-1)')
    
    # Model settings
    parser.add_argument('--model', type=str, default='gpt-5-mini',
                        help='OpenAI model to use (default: gpt-5-mini)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (default: from OPENAI_API_KEY env var)')
    
    # Processing settings
    parser.add_argument('--rate-limit-delay', type=float, default=0.5,
                        help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N entries (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save results every N entries (default: 100)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-evaluation of all entries')
    
    # Directories
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoint files (default: checkpoints)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files (default: logs)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.worker_id, args.log_dir)
    
    # Generate tmux commands mode
    if args.generate_tmux:
        if not args.dataset:
            parser.error("--generate-tmux requires --dataset")
        commands = generate_tmux_commands(
            args.dataset, args.output, args.num_workers,
            model=args.model, rate_limit_delay=args.rate_limit_delay
        )
        print(commands)
        return
    
    # Merge mode
    if args.merge:
        merge_worker_results(args.output, args.num_workers, 
                           output_format='jsonl', logger=logger)
        return
    
    # Split-only mode
    if args.split_only:
        if not args.dataset:
            parser.error("--split-only requires --dataset")
        split_dataset_for_workers(args.dataset, args.num_workers, 
                                 output_dir="worker_splits", logger=logger)
        return
    
    # Evaluation mode
    if not args.dataset:
        parser.error("--dataset is required for evaluation")
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
    
    # Load dataset
    df = load_dataset(args.dataset, logger)
    
    # If using workers, select the appropriate chunk
    if args.num_workers > 1:
        if args.worker_id is None:
            parser.error("--worker-id required when using --num-workers > 1")
        
        total = len(df)
        chunk_size = total // args.num_workers
        start_idx = args.worker_id * chunk_size
        end_idx = start_idx + chunk_size if args.worker_id < args.num_workers - 1 else total
        
        df = df.iloc[start_idx:end_idx].copy()
        logger.info(f"Worker {args.worker_id}: Processing entries {start_idx} to {end_idx-1} ({len(df)} entries)")
    
    # Initialize evaluator
    evaluator = LuxembourgishDatasetEvaluator(
        api_key=api_key,
        model=args.model,
        checkpoint_dir=args.checkpoint_dir,
        worker_id=args.worker_id,
        logger=logger
    )
    
    # Determine output path
    output_path = args.output
    if not any(output_path.endswith(ext) for ext in ['.csv', '.json', '.jsonl']):
        output_path = f"{output_path}.jsonl"
    
    # Run evaluation
    evaluator.evaluate_dataset(
        df,
        output_path=output_path,
        checkpoint_interval=args.checkpoint_interval,
        save_interval=args.save_interval,
        rate_limit_delay=args.rate_limit_delay,
        force_reevaluate=args.force
    )


if __name__ == "__main__":
    main()