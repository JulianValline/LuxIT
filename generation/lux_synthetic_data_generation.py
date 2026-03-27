# -*- coding: utf-8 -*-
"""
Enhanced Luxembourgish Synthetic Data Generator using DeepSeek R1
Supports both RTL news articles and Wikipedia articles as seed data.

Dependencies:
pip install openai pandas tqdm datasets
"""

import json
import os
import time
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Literal
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
from dotenv import load_dotenv
import traceback

import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from openai import OpenAI


# ==================== Data Source Types ====================
class DataSource(Enum):
    """Enumeration of supported data source types."""
    RTL = "rtl"
    WIKIPEDIA = "wikipedia"


# ==================== Configuration ====================
@dataclass
class SourceConfig:
    """Configuration for a specific data source."""
    input_file: str
    responses_output_file: str
    final_json_output: str
    final_csv_output: str
    generation_state_file: str
    source_type: DataSource
    
    def resolve_paths(self, base_path: str):
        """Convert relative paths to absolute paths."""
        self.input_file = os.path.join(base_path, self.input_file)
        self.responses_output_file = os.path.join(base_path, self.responses_output_file)
        self.final_json_output = os.path.join(base_path, self.final_json_output)
        self.final_csv_output = os.path.join(base_path, self.final_csv_output)
        self.generation_state_file = os.path.join(base_path, self.generation_state_file)


@dataclass
class Config:
    """Configuration settings for the data generator."""
    # Base path for all files
    base_path: str = './'
    
    # RTL source configuration
    rtl_config: SourceConfig = field(default_factory=lambda: SourceConfig(
        input_file='Data/filtered_rtl_articles.json',
        responses_output_file='model_responses_rtl.jsonl',
        final_json_output='instruction_answers_rtl.json',
        final_csv_output='instruction_answers_rtl.csv',
        generation_state_file='generation_state_rtl.json',
        source_type=DataSource.RTL
    ))
    
    # Wikipedia source configuration
    wiki_config: SourceConfig = field(default_factory=lambda: SourceConfig(
        input_file='Data/filtered_wiki_articles.json',
        responses_output_file='model_responses_wiki.jsonl',
        final_json_output='instruction_answers_wiki.json',
        final_csv_output='instruction_answers_wiki.csv',
        generation_state_file='generation_state_wiki.json',
        source_type=DataSource.WIKIPEDIA
    ))
    
    # Combined output configuration
    combined_output_json: str = 'instruction_tuning_dataset_all_sources.json'
    combined_output_csv: str = 'instruction_tuning_dataset_all_sources.csv'
    
    # Logging
    log_file: str = 'synthetic_data_generation.log'

    # loads the variables from .env into the environment
    load_dotenv()
    
    # API Settings
    api_key: str = os.environ.get("NEBIUS_API_KEY", "")
    api_base_url: str = "https://api.studio.nebius.com/v1/"
    model_name: str = "deepseek-ai/DeepSeek-R1-0528"
    max_tokens: int = 64000
    
    # Generation Settings
    num_pairs_per_article: int = 3  # Number of instruction-response pairs to generate per article
    max_samples_per_source: Optional[int] = None  # Default Limit for all sources
    max_rtl_samples: Optional[int] = 66776  # Specific limit for RTL articles
    max_wiki_samples: Optional[int] = None  # Specific limit for Wikipedia articles
    rate_limit_delay: float = 1.0  # seconds between API calls
    max_retries: int = 3
    retry_delay: float = 2.0  # exponential backoff base
    
    def __post_init__(self):
        """Resolve all paths and create necessary directories."""
        # Resolve paths for each source configuration
        self.rtl_config.resolve_paths(self.base_path)
        self.wiki_config.resolve_paths(self.base_path)
        
        # Resolve other paths
        self.combined_output_json = os.path.join(self.base_path, self.combined_output_json)
        self.combined_output_csv = os.path.join(self.base_path, self.combined_output_csv)
        self.log_file = os.path.join(self.base_path, self.log_file)
        
        # Create base directory if it doesn't exist
        Path(self.base_path).mkdir(parents=True, exist_ok=True)


# ==================== Logging Setup ====================
def setup_logging(log_file: str) -> logging.Logger:
    """
    Set up logging to both file and console with different levels.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('LuxDataGenerator')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - less verbose
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==================== Prompt Templates ====================
INSTRUCTION_PROMPT_TEMPLATE = """You are an expert in the Luxembourgish language tasked with creating high-quality synthetic training data for language models.

OBJECTIVE:
Generate {num_pairs} instruction-response pair(s) in authentic Luxembourgish based on the provided text. These pairs will be used for instruction fine-tuning of language models.

REQUIREMENTS:
1. LANGUAGE: All content MUST be in fluent, natural Luxembourgish
   - Use proper Luxembourgish grammar, spelling, and idioms
   - Avoid unnecessary German or French loan words
   - Ensure the language sounds natural to native speakers

2. QUALITY STANDARDS:
   - Instructions should be clear, specific, and answerable based on the provided text
   - Responses should be comprehensive, accurate, and well-structured
   - Include ALL necessary context in the instruction for a complete answer
   - If insufficient information exists, indicate that more details are needed

3. SUMMARIZATION INSTRUCTIONS:
   - When creating summarization tasks, ALWAYS include the original seed-text unchanged in the instruction for reference
   - The instruction should present the source text and ask for a summary
   - This ensures the training data contains both the source material and the summary

4. TEMPORAL CONTEXT:
   - When a date is provided, incorporate it appropriately
   - Add temporal context to maintain relevance when applicable
   - Consider whether dates belong in the instruction, response, or both

5. DIVERSITY:
   - Create varied types of instructions (e.g., summarization, Q&A, information extraction, explanation)
   - Vary complexity levels appropriately
   - Ensure each pair is unique and adds value

6. OUTPUT FORMAT:
   Return ONLY a valid JSON array with the following structure:
   [
     {{
       "instruction": "Clear instruction in Luxembourgish",
       "response": "Detailed response in Luxembourgish"
     }}
   ]

SOURCE TEXT:
{source_context}

Generate {num_pairs} high-quality instruction-response pair(s) based on the above text."""


# ==================== Core Classes ====================
class LuxembourgishDataGenerator:
    """Main class for generating synthetic Luxembourgish training data from multiple sources."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize the OpenAI client."""
        if not self.config.api_key:
            self.logger.warning("No API key provided.")
            return None
        
        try:
            client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key
            )
            self.logger.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def format_date(self, date_string: str) -> str:
        """
        Format date string, handling invalid dates gracefully.
        
        Args:
            date_string: Date string in format 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            Formatted date or placeholder for invalid dates
        """
        if not date_string or date_string == '0000-00-00 00:00:00':
            return None
        
        try:
            datetime_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
            return datetime_obj.strftime('%Y-%m-%d')
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Date parsing error for '{date_string}': {e}")
            return None
    
    def create_prompt(self, article: Dict[str, Any], source_type: DataSource) -> str:
        """
        Create an optimized prompt for the LLM based on the article type.
        
        Args:
            article: Dictionary containing article data
            source_type: Type of the data source (RTL or Wikipedia)
            
        Returns:
            Formatted prompt string
        """
        if source_type == DataSource.RTL:
            # RTL articles have date, title, header, and text
            date = self.format_date(article.get("date", ""))
            source_context = []
            
            if date:
                source_context.append(f"Article Date: {date}")
            
            source_context.extend([
                f"Title: {article.get('title', '')}",
                f"Header: {article.get('header', '')}",
                f"Content: {article.get('text', '')}"
            ])
            
            source_context_str = "\n".join(source_context)
            
        elif source_type == DataSource.WIKIPEDIA:
            # Wikipedia articles have title and text
            source_context_str = "\n".join([
                f"Wikipedia Article Title: {article.get('title', '')}",
                f"Article Content: {article.get('text', '')}"
            ])
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return INSTRUCTION_PROMPT_TEMPLATE.format(
            num_pairs=self.config.num_pairs_per_article,
            source_context=source_context_str
        )
    
    def clean_response(self, response: str) -> str:
        """
        Remove thinking tags and clean the LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned response text
        """
        # Remove <think>...</think> blocks
        think_pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
        cleaned = think_pattern.sub('', response).strip()
        
        # Remove any markdown code block markers if present
        cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
        
        return cleaned
    
    def call_api_with_retry(self, article: Dict[str, Any], source_type: DataSource) -> Optional[str]:
        """
        Call the API with retry logic and error handling.
        
        Args:
            article: Article dictionary
            source_type: Type of the data source
            
        Returns:
            Cleaned API response or None if failed
        """
        if not self.client:
            self.logger.warning("No API client available - skipping API call")
            return None
        
        prompt = self.create_prompt(article, source_type)
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{self.config.max_retries}")
                
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=0.7,
                    stream=False
                )
                
                raw_response = completion.choices[0].message.content
                cleaned_response = self.clean_response(raw_response)
                
                self.logger.debug(f"API call successful, response length: {len(cleaned_response)}")
                
                return cleaned_response
                
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.debug(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    article_id = article.get('article_id') or article.get('id') or 'unknown'
                    self.logger.error(f"All API call attempts failed for article {article_id}")
                    return None
    
    def extract_json_pairs(self, text: str) -> List[Dict[str, str]]:
        """
        Extract instruction-response pairs from model output.
        
        Args:
            text: Model output text
            
        Returns:
            List of instruction-response dictionaries
        """
        if not text:
            return []
        
        # Try to find JSON content
        json_match = re.search(r'[\[\{].*[\]\}]', text, re.DOTALL)
        if not json_match:
            self.logger.debug("No JSON structure found in response")
            return []
        
        json_str = json_match.group()
        
        try:
            data = json.loads(json_str)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return self._validate_pairs(data)
            elif isinstance(data, dict):
                # Check if it's a single pair
                if "instruction" in data and "response" in data:
                    return self._validate_pairs([data])
                # Check if it contains a list
                for value in data.values():
                    if isinstance(value, list):
                        return self._validate_pairs(value)
            
            self.logger.debug("Unexpected JSON structure")
            return []
            
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON parsing error: {e}")
            return []
    
    def _validate_pairs(self, pairs: List[Dict]) -> List[Dict[str, str]]:
        """
        Validate and clean instruction-response pairs.
        
        Args:
            pairs: List of potential instruction-response pairs
            
        Returns:
            List of valid pairs
        """
        valid_pairs = []
        
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            
            # Handle typos in keys
            if "responsive" in pair and "response" not in pair:
                pair["response"] = pair.pop("responsive")
            
            if "instruction" in pair and "response" in pair:
                # Clean up the text
                pair["instruction"] = pair["instruction"].strip()
                pair["response"] = pair["response"].strip()
                
                # Only add non-empty pairs
                if pair["instruction"] and pair["response"]:
                    valid_pairs.append(pair)
        
        return valid_pairs


class SourceProcessor:
    """Handles processing for a specific data source."""
    
    def __init__(self, source_config: SourceConfig, generator: LuxembourgishDataGenerator, 
                 logger: logging.Logger, max_samples: Optional[int] = None):
        """
        Initialize the source processor.
        
        Args:
            source_config: Configuration for this source
            generator: The data generator instance
            logger: Logger instance
            max_samples: Maximum samples to process for this source
        """
        self.source_config = source_config
        self.generator = generator
        self.logger = logger
        self.max_samples = max_samples
        self.generation_state = self._load_generation_state()
        self.processed_ids = set()
        self.statistics = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'api_errors': 0,
            'parsing_errors': 0,
            'session_processed': 0,
            'cumulative_processed': 0
        }
    
    def _load_generation_state(self) -> Dict[str, Any]:
        """
        Load the generation state for this source.
        
        Returns:
            Dictionary containing the generation state
        """
        if os.path.exists(self.source_config.generation_state_file):
            try:
                with open(self.source_config.generation_state_file, 'r') as f:
                    state = json.load(f)
                    self.logger.info(
                        f"[{self.source_config.source_type.value}] Loaded state: "
                        f"{state['total_articles_processed']} articles processed"
                    )
                    return state
            except Exception as e:
                self.logger.warning(f"Could not load generation state: {e}")
                return self._create_fresh_state()
        return self._create_fresh_state()
    
    def _create_fresh_state(self) -> Dict[str, Any]:
        """Create a fresh generation state."""
        return {
            'source_type': self.source_config.source_type.value,
            'total_articles_processed': 0,
            'last_processed_index': -1,
            'last_processed_id': None,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'sessions': []
        }
    
    def _save_generation_state(self):
        """Save the current generation state."""
        self.generation_state['last_updated'] = datetime.now().isoformat()
        with open(self.source_config.generation_state_file, 'w') as f:
            json.dump(self.generation_state, f, indent=2)
    
    def load_checkpoint(self) -> Tuple[set, List[str]]:
        """
        Load processing checkpoint if it exists.
        
        Returns:
            Tuple of (processed_ids, existing_responses)
        """
        processed_ids = set()
        responses = []
        
        if os.path.exists(self.source_config.responses_output_file):
            self.logger.info(f"Loading checkpoint from {self.source_config.responses_output_file}")
            
            with open(self.source_config.responses_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        processed_ids.add(str(result['source_id']))
                        responses.append(result['synthetic_text'])
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.debug(f"Skipping malformed checkpoint line: {e}")
            
            self.logger.info(f"Resumed: {len(processed_ids)} articles already processed")
        
        return processed_ids, responses
    
    def get_article_id(self, article: Dict[str, Any]) -> str:
        """
        Get the article ID based on source type.

        Args:
            article: source article as dictionary

        Returns:
            Article ID as string or "unknown" if not found
        """
        if self.source_config.source_type == DataSource.RTL:
            return str(article.get('article_id', 'unknown'))
        elif self.source_config.source_type == DataSource.WIKIPEDIA:
            return str(article.get('id', 'unknown'))
        return 'unknown'
    
    def generate_synthetic_data(self, source_data: List[Dict]) -> List[str]:
        """
        Generate synthetic data from source articles with sample limit support.
        
        Args:
            source_data: List of source articles
            
        Returns:
            List of model responses
        """
        processed_ids, model_responses = self.load_checkpoint()
        self.processed_ids = processed_ids
        
        # Start from where we left off
        start_index = self.generation_state['last_processed_index'] + 1
        
        # Determine how many samples to process
        if self.max_samples is not None:
            # Calculate remaining samples to reach max_samples
            already_processed = self.generation_state['total_articles_processed']
            remaining_to_process = self.max_samples - already_processed
            
            if remaining_to_process <= 0:
                self.logger.info(
                    f"[{self.source_config.source_type.value}] Already processed "
                    f"{already_processed}/{self.max_samples} articles"
                )
                return model_responses
            
            # Process only the remaining amount
            end_index = min(start_index + remaining_to_process, len(source_data))
            articles_to_process = source_data[start_index:end_index]

        else:
            # Process all remaining articles
            articles_to_process = source_data[start_index:]
        
        if not articles_to_process:
            self.logger.info(f"[{self.source_config.source_type.value}] No new articles to process")
            return model_responses
        
        self.logger.info(
            f"[{self.source_config.source_type.value}] Processing {len(articles_to_process)} "
            f"articles (indices {start_index} to {start_index + len(articles_to_process) - 1})"
        )
        
        # Track session start
        session_info = {
            'start_time': datetime.now().isoformat(),
            'start_index': start_index,
            'planned_count': len(articles_to_process)
        }
        
        with open(self.source_config.responses_output_file, 'a', encoding='utf-8') as f:
            progress_bar = tqdm(
                articles_to_process, 
                desc=f"Generating {self.source_config.source_type.value} data"
            )
            
            for idx, article in enumerate(progress_bar):
                article_id = self.get_article_id(article)
                global_index = start_index + idx  # Track position in original dataset
                
                if article_id in self.processed_ids:
                    self.logger.debug(f"Skipping already processed article: {article_id}")
                    continue
                
                self.logger.info(f"Processing article ID: {article_id} (index: {global_index})")
                self.statistics['total_processed'] += 1
                self.statistics['session_processed'] += 1
                
                # Generate synthetic data
                synthetic_data = self.generator.call_api_with_retry(
                    article, 
                    self.source_config.source_type
                )
                
                if synthetic_data:
                    model_responses.append(synthetic_data)
                    self.statistics['successful'] += 1
                    
                    # Save checkpoint
                    result = {
                        'source_id': article_id,
                        'source_title': article.get('title', ''),
                        'source_type': self.source_config.source_type.value,
                        'synthetic_text': synthetic_data,
                        'global_index': global_index,
                        'timestamp': time.time()
                    }
                    
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()  # Ensure data is written
                    
                    self.processed_ids.add(article_id)
                    
                    # Update generation state
                    self.generation_state['total_articles_processed'] += 1
                    self.generation_state['last_processed_index'] = global_index
                    self.generation_state['last_processed_id'] = article_id
                    
                    # Save state periodically (every 5 articles)
                    if self.statistics['session_processed'] % 5 == 0:
                        self._save_generation_state()
                else:
                    self.statistics['failed'] += 1
                    self.statistics['api_errors'] += 1
                
                # Rate limiting
                time.sleep(self.generator.config.rate_limit_delay)
                
                # Check if we've reached the limit
                if self.max_samples and self.generation_state['total_articles_processed'] >= self.max_samples:
                    self.logger.info(
                        f"[{self.source_config.source_type.value}] Reached limit of {self.max_samples}"
                    )
                    break
        
        # Final state save
        session_info['end_time'] = datetime.now().isoformat()
        session_info['articles_processed'] = self.statistics['session_processed']
        session_info['end_index'] = self.generation_state['last_processed_index']
        self.generation_state['sessions'].append(session_info)
        self._save_generation_state()
        
        self.statistics['cumulative_processed'] = self.generation_state['total_articles_processed']
        
        return model_responses
    
    def process_and_save_results(self, model_responses: List[str]) -> pd.DataFrame:
        """
        Process model responses and save results in ShareGPT format.
        
        Args:
            model_responses: List of raw model responses
        """
        self.logger.info(f"[{self.source_config.source_type.value}] Processing responses...")
        
        combined_data = []
        empty_count = 0
        
        for i, response in enumerate(model_responses):
            if not response:
                empty_count += 1
                continue
            
            pairs = self.generator.extract_json_pairs(response)
            if pairs:
                # Add source type metadata
                for pair in pairs:
                    pair['source_type'] = self.source_config.source_type.value
                combined_data.extend(pairs)
                self.logger.debug(f"Extracted {len(pairs)} pairs from response {i}")
            else:
                self.statistics['parsing_errors'] += 1
        
        self.logger.info(
            f"[{self.source_config.source_type.value}] Extracted {len(combined_data)} pairs "
            f"({empty_count} empty responses)"
        )
        
        # Add ShareGPT format
        for item in combined_data:
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            
            conversations = [
                {
                    "from": "human",
                    "value": instruction
                },
                {
                    "from": "gpt", 
                    "value": response
                }
            ]
            item["conversations"] = conversations
        
        # Save as JSON
        self.logger.info(f"Saving JSON to {self.source_config.final_json_output}")
        with open(self.source_config.final_json_output, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        df = pd.DataFrame(combined_data)
        if not df.empty:
            df.to_csv(self.source_config.final_csv_output, index=False, encoding='utf-8')
            self.logger.info(f"Saved CSV to {self.source_config.final_csv_output}")
        
        return df


# ==================== Main Execution ====================
def process_source(
    source_type: DataSource,
    config: Config,
    logger: logging.Logger,
    generator: LuxembourgishDataGenerator
) -> Optional[pd.DataFrame]:
    """
    Process a single data source.
    
    Args:
        source_type: Type of source to process
        config: Configuration object
        logger: Logger instance
        generator: Data generator instance
        
    Returns:
        DataFrame with results or None if no processing occurred
    """
    # Select appropriate source configuration
    if source_type == DataSource.RTL:
        source_config = config.rtl_config
        # Use specific RTL limit if set, otherwise use general limit
        max_samples = config.max_rtl_samples or config.max_samples_per_source
    elif source_type == DataSource.WIKIPEDIA:
        source_config = config.wiki_config
        # Use specific Wiki limit if set, otherwise use general limit
        max_samples = config.max_wiki_samples or config.max_samples_per_source
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    
    # Check if input file exists
    if not os.path.exists(source_config.input_file):
        logger.warning(f"Input file not found: {source_config.input_file}")
        logger.info(f"Skipping {source_type.value} processing")
        return None
    
    # Load source data
    logger.info(f"Loading {source_type.value} data from {source_config.input_file}")
    with open(source_config.input_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    logger.info(f"Loaded {len(source_data)} {source_type.value} articles")
    
    # Initialize processor
    processor = SourceProcessor(source_config, generator, logger, max_samples)
    
    # Check current progress
    already_processed = processor.generation_state['total_articles_processed']
    if already_processed > 0:
        logger.info(
            f"[{source_type.value}] Previous progress: {already_processed} articles processed"
        )
    
    # Generate synthetic data
    model_responses = processor.generate_synthetic_data(source_data)
    
    # Process and save results
    if processor.statistics['session_processed'] > 0:
        df = processor.process_and_save_results(model_responses)
        
        # Log statistics
        logger.info(f"[{source_type.value}] Session Statistics:")
        logger.info(f"  Processed this session: {processor.statistics['session_processed']}")
        logger.info(f"  Successful: {processor.statistics['successful']}")
        logger.info(f"  Failed: {processor.statistics['failed']}")
        logger.info(f"  Total processed: {processor.statistics['cumulative_processed']}")
        
        return df
    else:
        logger.info(f"[{source_type.value}] No new articles processed")
        # Return existing data if available
        if os.path.exists(source_config.final_csv_output):
            return pd.read_csv(source_config.final_csv_output)
        return None


def combine_datasets(
    datasets: Dict[DataSource, pd.DataFrame],
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Combine datasets from multiple sources into a single dataset.
    
    Args:
        datasets: Dictionary mapping source types to their DataFrames
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Combined DataFrame
    """
    if not datasets:
        logger.warning("No datasets to combine")
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(datasets.values(), ignore_index=True)
    
    # Save combined dataset
    logger.info(f"Saving combined dataset with {len(combined_df)} total pairs")
    
    # Save as JSON
    combined_data = combined_df.to_dict('records')
    with open(config.combined_output_json, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved combined JSON to {config.combined_output_json}")
    
    # Save as CSV
    combined_df.to_csv(config.combined_output_csv, index=False, encoding='utf-8')
    logger.info(f"Saved combined CSV to {config.combined_output_csv}")
    
    return combined_df

def combine_existing_datasets(config: Config = None):
    """
    Combine existing RTL and Wikipedia datasets into a single dataset.
    Call this after running parallel generation sessions.
    """
    if config is None:
        config = Config()
    
    logger = setup_logging(config.log_file)
    logger.info("Combining existing datasets...")
    
    datasets = {}
    
    # Load RTL data if exists
    if os.path.exists(config.rtl_config.final_csv_output):
        rtl_df = pd.read_csv(config.rtl_config.final_csv_output)
        datasets[DataSource.RTL] = rtl_df
        logger.info(f"Loaded RTL data: {len(rtl_df)} pairs")
    
    # Load Wikipedia data if exists
    if os.path.exists(config.wiki_config.final_csv_output):
        wiki_df = pd.read_csv(config.wiki_config.final_csv_output)
        datasets[DataSource.WIKIPEDIA] = wiki_df
        logger.info(f"Loaded Wikipedia data: {len(wiki_df)} pairs")
    
    if len(datasets) > 1:
        combined_df = combine_datasets(datasets, config, logger)
        print(f"✅ Combined {len(combined_df)} total pairs")
        print(f"📁 Output: {config.combined_output_json}")
        return combined_df
    else:
        logger.warning("Need at least 2 source datasets to combine")
        return None


def main(
    sources: Optional[List[DataSource]] = None,
    max_samples_per_source: Optional[int] = None,
    max_rtl_samples: Optional[int] = None,
    max_wiki_samples: Optional[int] = None
):
    """
    Main execution function with configurable sources and sample limits.
    
    Args:
        sources: List of sources to process (None = all available)
        max_samples_per_source: Maximum samples per source (None = unlimited)
        max_rtl_samples: Specific limit for RTL articles
        max_wiki_samples: Specific limit for Wikipedia articles
    """
    # Initialize configuration
    config = Config()
    
    # Override max_samples if provided
    if max_samples_per_source is not None:
        config.max_samples_per_source = max_samples_per_source
    if max_rtl_samples is not None:
        config.max_rtl_samples = max_rtl_samples
    if max_wiki_samples is not None:
        config.max_wiki_samples = max_wiki_samples
    
    # Determine which sources to process
    if sources is None:
        sources = list(DataSource)  # Process all sources
    
    # Set up logging
    logger = setup_logging(config.log_file)
    logger.info("="*60)
    logger.info("Starting Luxembourgish Synthetic Data Generation")
    logger.info(f"Sources to process: {[s.value for s in sources]}")

    # Log the limits for each source
    for source in sources:
        if source == DataSource.RTL:
            limit = config.max_rtl_samples or config.max_samples_per_source
            logger.info(f"RTL limit: {limit if limit else 'unlimited'}")
        elif source == DataSource.WIKIPEDIA:
            limit = config.max_wiki_samples or config.max_samples_per_source
            logger.info(f"Wikipedia limit: {limit if limit else 'unlimited'}")

    logger.info("="*60)
    
    try:
        # Initialize generator
        generator = LuxembourgishDataGenerator(config, logger)
        
        # Process each source
        datasets = {}
        for source_type in sources:
            logger.info(f"\nProcessing {source_type.value} source...")
            logger.info("-" * 40)
            
            df = process_source(source_type, config, logger, generator)
            if df is not None and not df.empty:
                datasets[source_type] = df
                logger.info(f"✓ {source_type.value}: {len(df)} instruction-response pairs")
        
        # Combine datasets if multiple sources were processed
        if len(datasets) > 1:
            logger.info("\nCombining datasets from all sources...")
            combined_df = combine_datasets(datasets, config, logger)
            
            # Create HuggingFace dataset
            dataset = Dataset.from_pandas(combined_df)
            logger.info(f"Created combined HuggingFace dataset with {len(dataset)} examples")
        elif datasets:
            # Single source processed
            source_type = list(datasets.keys())[0]
            logger.info(f"\nSingle source processed: {source_type.value}")
            dataset = Dataset.from_pandas(list(datasets.values())[0])
        else:
            logger.warning("No data was processed from any source")
            dataset = None
        
        # Final summary
        logger.info("="*60)
        logger.info("Generation Complete - Summary")
        logger.info("="*60)
        
        total_pairs = sum(len(df) for df in datasets.values())
        logger.info(f"Total instruction-response pairs generated: {total_pairs}")
        
        for source_type, df in datasets.items():
            logger.info(f"  {source_type.value}: {len(df)} pairs")
        
        print("\n✅ Script completed successfully!")
        print(f"📊 Generated {total_pairs} total instruction-response pairs")
        print("\n📁 Output files:")
        
        for source_type in sources:
            if source_type == DataSource.RTL:
                source_config = config.rtl_config
            else:
                source_config = config.wiki_config
            
            if os.path.exists(source_config.final_json_output):
                print(f"\n  {source_type.value} outputs:")
                print(f"    - JSON: {source_config.final_json_output}")
                print(f"    - CSV: {source_config.final_csv_output}")
        
        if len(datasets) > 1:
            print(f"\n  Combined outputs:")
            print(f"    - JSON: {config.combined_output_json}")
            print(f"    - CSV: {config.combined_output_csv}")
        
        print(f"\n  Log file: {config.log_file}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        print(f"\n❌ Unexpected error occurred. Check {config.log_file} for details.")
        raise


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print("Usage: python script.py [options]")
        print("\nOptions:")
        print("  --sources rtl,wiki     Specify sources to process (comma-separated)")
        print("  --max-samples N        Maximum samples per source")
        print("  --max-rtl N            Maximum samples for RTL articles")
        print("  --max-wiki N           Maximum samples for Wikipedia articles")
        print("  --rtl-only             Process only RTL articles")
        print("  --wiki-only            Process only Wikipedia articles")
        print("  --combine-only         Combine existing RTL and Wiki datasets")
        print("\nExamples:")
        print("  python LuxSyntheticDataGeneration.py --max-samples 100")
        print("  python LuxSyntheticDataGeneration.py --max-rtl 6 --max-wiki 4")
        print("  python LuxSyntheticDataGeneration.py --sources rtl --max-samples 50")
        print("  python LuxSyntheticDataGeneration.py --wiki-only --max-wiki 10")
        print("  python LuxSyntheticDataGeneration.py --combine-only")
        sys.exit(0)
    
    
    if '--combine-only' in args:
        combine_existing_datasets()
        sys.exit(0)
    
    # Parse sources
    sources = None
    if '--rtl-only' in args:
        sources = [DataSource.RTL]
    elif '--wiki-only' in args:
        sources = [DataSource.WIKIPEDIA]
    elif '--sources' in args:
        idx = args.index('--sources')
        if idx + 1 < len(args):
            source_names = args[idx + 1].split(',')
            sources = []
            for name in source_names:
                if name.lower() == 'rtl':
                    sources.append(DataSource.RTL)
                elif name.lower() in ['wiki', 'wikipedia']:
                    sources.append(DataSource.WIKIPEDIA)
    
    # Parse max samples
    max_samples = None
    if '--max-samples' in args:
        idx = args.index('--max-samples')
        if idx + 1 < len(args):
            try:
                max_samples = int(args[idx + 1])
            except ValueError:
                print("Error: --max-samples must be followed by a number")
                sys.exit(1)

    # Parse RTL-specific limit
    max_rtl = None
    if '--max-rtl' in args:
        idx = args.index('--max-rtl')
        if idx + 1 < len(args):
            try:
                max_rtl = int(args[idx + 1])
            except ValueError:
                print("Error: --max-rtl must be followed by a number")
                sys.exit(1)
    
    # Parse Wikipedia-specific limit
    max_wiki = None
    if '--max-wiki' in args:
        idx = args.index('--max-wiki')
        if idx + 1 < len(args):
            try:
                max_wiki = int(args[idx + 1])
            except ValueError:
                print("Error: --max-wiki must be followed by a number")
                sys.exit(1)
    
    # Run main
    main(
        sources=sources,
        max_samples_per_source=max_samples,
        max_rtl_samples=max_rtl,
        max_wiki_samples=max_wiki
    )