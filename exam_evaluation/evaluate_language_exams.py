"""
Luxembourgish Language Exam Evaluation Pipeline using Unsloth

This script evaluates language models on Luxembourgish language exams (A1-C2)
using Unsloth for model loading, supporting both base models and LoRA adapters.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from random import shuffle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from Levenshtein import distance
import torch
import regex as re

from unsloth import FastLanguageModel
from unsloth import FastModel  # For Gemma models

# Configure logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging to both file and console."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"
    
    # Create a custom logger (don't use root logger which other libraries may have configured)
    logger = logging.getLogger("luxembourgish_eval")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates on re-runs
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str                                      # Display name for the model
    base_model: str                                # HuggingFace model ID or path
    lora_path: Optional[str] = None                # Path to LoRA adapter (None for base model only)
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[torch.dtype] = None
    # Chat template options:
    # - None: Auto-detect (use LoRA tokenizer if available, else base model)
    # - "path/to/tokenizer": Load tokenizer from this path
    # - A chat template string: Use exact template
    chat_template: Optional[str] = None
    enable_thinking: Optional[bool] = None                   # Default to False for standard evaluation


@dataclass
class ExamConfig:
    """Configuration for a language exam."""
    level: str                                     # e.g., "a1", "a2", "b1", "b2", "c1", "c2"
    test_file: str                                 # Path to the test TSV file
    ground_truth_file: str                         # Path to the ground truth TSV file

# Prompt that will be passed as the user instruction
SYSTEM_INSTRUCTION = """Ech ginn dir e Sproochentest fir Lëtzebuergesch. Fir all Deel kriss du en TEXT wou en Deel feelt, markéiert mat [BLANK], an eng Lëscht mat méiglechen ÄNTWERTEN, wou all Optioun mat engem Komma getrennt ass. Wiel déi Optioun, déi am beschten an d'Plaz vum [BLANK] passt. Gëff JUST déi richteg Optioun als Äntwert. Keng Erklärungen."""

class UnslothModelWrapper:
    """Wrapper for loading and running inference with Unsloth models."""
    
    # Model families that use FastModel instead of FastLanguageModel
    FAST_MODEL_FAMILIES = ["gemma"]
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.model_class = None
    
    def _detect_model_class(self):
        """Detect whether to use FastModel or FastLanguageModel based on model name."""
        model_name_lower = self.config.base_model.lower()
        
        for family in self.FAST_MODEL_FAMILIES:
            if family in model_name_lower:
                self.logger.info(f"  Detected {family} model family -> using FastModel")
                return FastModel
        
        self.logger.info("  Using FastLanguageModel")
        return FastLanguageModel
    
    def _check_tokenizer_exists(self, path: str) -> bool:
        """Check if a tokenizer exists at the given path."""
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
        path = Path(path)
        return any((path / f).exists() for f in tokenizer_files)
    
    def _load_tokenizer_from_path(self, path: str):
        """Load tokenizer from a specific path."""
        from transformers import AutoTokenizer
        self.logger.info(f"  Loading tokenizer from: {path}")
        return AutoTokenizer.from_pretrained(path)
    
    def _apply_chat_template_override(self):
        """Apply custom chat template if specified in config."""
        if self.config.chat_template is None:
            return
        
        # Check if it's a path to a tokenizer or a template string
        if os.path.isdir(self.config.chat_template):
            # It's a path to a tokenizer directory
            self.logger.info(f"  Loading chat template from tokenizer at: {self.config.chat_template}")
            template_tokenizer = self._load_tokenizer_from_path(self.config.chat_template)
            if hasattr(template_tokenizer, 'chat_template') and template_tokenizer.chat_template:
                self.tokenizer.chat_template = template_tokenizer.chat_template
                self.logger.info("  Chat template loaded from external tokenizer")
            else:
                self.logger.warning("  No chat template found in specified tokenizer path")
        else:
            # It's a template string
            self.logger.info("  Applying custom chat template string from config")
            self.tokenizer.chat_template = self.config.chat_template
    
    def _log_chat_template_info(self):
        """Log information about the current chat template."""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            template = self.tokenizer.chat_template
            # Show first 300 chars of the template for identification
            template_preview = template[:300].replace('\n', '\\n')
            self.logger.info(f"  Chat template in use ({len(template)} chars):")
            self.logger.info(f"    Preview: {template_preview}...")
        else:
            self.logger.warning("  WARNING: No chat template found in tokenizer!")
            self.logger.warning("  The model may not format prompts correctly.")
        
    def load(self):
        """Load the model with optional LoRA adapter."""
        self.logger.info(f"Loading model: {self.config.name}")
        self.logger.info(f"  Base model: {self.config.base_model}")
        
        # Detect which model class to use
        self.model_class = self._detect_model_class()
        
        if self.config.lora_path:
            self.logger.info(f"  LoRA adapter: {self.config.lora_path}")
            
            # Check if tokenizer exists in LoRA directory
            lora_has_tokenizer = self._check_tokenizer_exists(self.config.lora_path)
            
            if lora_has_tokenizer:
                self.logger.info("  Found tokenizer in LoRA directory")
            else:
                self.logger.warning("  No tokenizer found in LoRA directory!")
                self.logger.warning("  Will use base model's tokenizer - chat template may not match training!")
            
            # Load model with LoRA adapter
            # Unsloth's from_pretrained can load directly from LoRA path
            # It reads adapter_config.json to find the base model
            self.model, self.tokenizer = self.model_class.from_pretrained(
                model_name=self.config.lora_path,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                dtype=self.config.dtype,
            )
            
            # If LoRA has its own tokenizer, explicitly load it to ensure chat template is correct
            # This is a safety measure in case Unsloth didn't load it from the LoRA directory
            if lora_has_tokenizer:
                from transformers import AutoTokenizer
                lora_tokenizer = AutoTokenizer.from_pretrained(self.config.lora_path)
                
                # Check if the chat templates differ
                lora_template = getattr(lora_tokenizer, 'chat_template', None)
                current_template = getattr(self.tokenizer, 'chat_template', None)
                
                if lora_template and lora_template != current_template:
                    self.logger.info("  Overriding tokenizer chat template with LoRA's chat template")
                    self.tokenizer.chat_template = lora_template
                elif lora_template:
                    self.logger.info("  Chat template already matches LoRA's tokenizer")
                else:
                    self.logger.warning("  LoRA tokenizer has no chat template!")
            
        else:
            self.logger.info("  Loading base model only (no LoRA)")
            self.model, self.tokenizer = self.model_class.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                dtype=self.config.dtype,
            )
        
        # Apply chat template override if specified (overrides everything above)
        self._apply_chat_template_override()
        
        # Log which chat template is being used
        self._log_chat_template_info()
        
        # Enable inference mode
        self.model_class.for_inference(self.model)
        self.logger.info("Model loaded successfully")
        
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response for the given prompt."""
        
        # Combine the Luxembourgish instruction with the specific question
        full_content = f"{SYSTEM_INSTRUCTION}\n\n{prompt}"

        # Format as a single user message
        messages = [
            {"role": "user", "content": full_content}
        ]
        
        # Prepare kwargs for chat template
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        
        # Only inject the argument if it is explicitly True or False in config.
        # If it is None, we skip this block, protecting Mistral/Llama.
        if self.config.enable_thinking is not None:
             chat_template_kwargs["enable_thinking"] = self.config.enable_thinking

        # Apply chat template with error handling for the kwargs
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                **chat_template_kwargs
            )
        except TypeError:
            # if tokenizer does not accept 'enable_thinking'
            if "enable_thinking" in chat_template_kwargs:
                # self.logger.warning("Tokenizer rejected enable_thinking arg. Retrying without it.")
                del chat_template_kwargs["enable_thinking"]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    **chat_template_kwargs
                )
            else:
                raise

        # Tokenize
        inputs = self.tokenizer(text=input_text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def unload(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            self.logger.info(f"Model {self.config.name} unloaded")


class LuxembourgishEvaluator:
    """Main evaluator class for Luxembourgish language exams."""
    
    def __init__(self, output_base_dir: str = "evaluation_results"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(str(self.output_base_dir))
        
    def parse_test_file(self, file_path: str) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
        """Parse a test TSV file and return tests and answer options."""
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()[1:]  # Skip header
        
        tests = {}
        answers = {}
        for idx, line in enumerate(lines):
            parts = line.strip().split("\t")
            tests[idx] = parts[0]
            answer_str = parts[1].replace("ANSWERS: [", "").replace("]", "")
            answers[idx] = [a.strip() for a in answer_str.split(",")]
        
        return tests, answers
    
    def parse_ground_truth(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Parse ground truth file and return categories and correct answers."""
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
        
        categories = [line.split("\t")[0] for line in lines[1:]]
        answers = [line.split("\t")[3].strip() for line in lines[1:]]
        
        return categories, answers
    
    def post_process(self, output: str, answers: List[str]) -> str:
        """Post-process model output to extract the most likely answer."""
        # Count occurrences of each answer in the output
        answer_frequencies = [output.count(answer) for answer in answers]
        
        # If frequencies are not all equal and there's a unique maximum, return that answer
        if not all(x == answer_frequencies[0] for x in answer_frequencies):
            if answer_frequencies.count(max(answer_frequencies)) == 1:
                return answers[answer_frequencies.index(max(answer_frequencies))]
        
        return output
    
    def find_closest_match(self, candidate: str, references: List[str], 
                           error_threshold: float = 0.5) -> str:
        """Find the closest matching answer using Levenshtein distance."""
        if candidate in references:
            return candidate
        
        levenshtein_distances = [distance(candidate, ref) for ref in references]
        minimum = min(levenshtein_distances)
        
        # If there are multiple closest matches, mark as invalid
        if levenshtein_distances.count(minimum) > 1:
            return "I"
        
        closest = references[levenshtein_distances.index(minimum)]
        relative_distance = minimum / max(len(candidate), 1)
        
        # Accept if within threshold or distance is 1
        if relative_distance <= error_threshold or minimum == 1:
            return closest
        
        return "I"
    
    def run_exam(self, model_wrapper: UnslothModelWrapper, 
                 tests: Dict[int, str], answers: Dict[int, List[str]]) -> Dict[int, str]:
        """Run a single exam and return raw outputs."""
        outputs = {}
        test_keys = list(tests.keys())
        shuffle(test_keys)
        
        for idx, key in enumerate(test_keys):
            self.logger.info(f"  Question {idx + 1}/{len(tests)}")
            
            input_prompt = f" INPUT:\n{tests[key]} ANSWERS: {answers[key]}\nOUTPUT:\n"
            
            # Generate response
            output = model_wrapper.generate(input_prompt)
            
            # Post-process
            output = self.post_process(output, answers[key])

            escaped_options = [re.escape(opt) for opt in answers[key]]
            options_pattern = f"({'|'.join(escaped_options)})"
            
            # all options in the text that appear as whole words
            pattern = f"\\b{options_pattern}\\b"
            matches = list(re.finditer(pattern, output, re.IGNORECASE))
            
            final_answer = ""
            if matches:
                final_answer = matches[-1].group(0)
            else:
                # take the first line if no valid option found
                if "\n" in output.strip():
                    final_answer = output.strip().split("\n")[0]
                else:
                    final_answer = output.strip()
            
            outputs[key] = final_answer.strip()
            self.logger.debug(f"    Q: {tests[key][:50]}... -> Raw: {output.strip()[:50]}... -> Parsed: {final_answer}")
        
        return outputs
    
    def format_outputs(self, raw_outputs: Dict[int, str], 
                       answers: Dict[int, List[str]]) -> Tuple[List[str], List[str]]:
        """Format raw outputs to match answer options and convert to letters."""
        letters = ["A", "B", "C", "D", "E", "F", "G"]
        formatted = []
        formatted_letters = []
        
        for idx in range(len(raw_outputs)):
            options = answers[idx]
            raw = raw_outputs[idx].replace("[", "").replace("]", "")
            
            # Find closest match
            matched = self.find_closest_match(raw, options)
            formatted.append(matched)
            
            # Convert to letter
            if matched == "I":
                formatted_letters.append("I")
            else:
                letter_idx = options.index(matched)
                formatted_letters.append(letters[letter_idx])
        
        return formatted, formatted_letters
    
    def calculate_scores(self, formatted_outputs: List[str], 
                         categories: List[str], 
                         ground_truth: List[str]) -> Dict[str, float]:
        """Calculate scores per category and total."""
        # Count samples per category
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Initialize results
        results = {"total": 0}
        for cat in category_counts:
            results[cat] = 0
        
        # Calculate correct answers
        for i, (pred, gt) in enumerate(zip(formatted_outputs, ground_truth)):
            correct = int(pred == gt)
            results["total"] += correct
            results[categories[i]] += correct
        
        # Convert to percentages
        results["total"] = results["total"] / len(ground_truth)
        for cat in category_counts:
            results[cat] = results[cat] / category_counts[cat]
        
        return results
    
    def evaluate_model_on_exam(self, model_wrapper: UnslothModelWrapper,
                                exam: ExamConfig, model_output_dir: Path) -> Dict[str, float]:
        """Evaluate a single model on a single exam."""
        self.logger.info(f"  Running exam: {exam.level.upper()}")
        
        # Create exam output directory
        exam_dir = model_output_dir / exam.level
        exam_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse test file
        tests, answers = self.parse_test_file(exam.test_file)
        self.logger.info(f"    Loaded {len(tests)} questions")
        
        # Run the exam
        raw_outputs = self.run_exam(model_wrapper, tests, answers)
        
        # Save raw outputs
        raw_output_file = exam_dir / "raw_outputs.txt"
        with open(raw_output_file, "w", encoding="utf-8") as f:
            for i in range(len(tests)):
                output = raw_outputs.get(i, "INVALID_OUTPUT")
                f.write(f"{output}\n")
        
        # Format outputs
        formatted, formatted_letters = self.format_outputs(raw_outputs, answers)
        
        # Save formatted outputs
        formatted_file = exam_dir / "formatted_outputs.txt"
        with open(formatted_file, "w", encoding="utf-8") as f:
            for line in formatted:
                f.write(f"{line}\n")
        
        formatted_letters_file = exam_dir / "formatted_letters.txt"
        with open(formatted_letters_file, "w", encoding="utf-8") as f:
            for line in formatted_letters:
                f.write(f"{line}\n")
        
        # Calculate scores
        categories, ground_truth = self.parse_ground_truth(exam.ground_truth_file)
        scores = self.calculate_scores(formatted, categories, ground_truth)
        
        # Save scores
        scores_file = exam_dir / "scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        
        self.logger.info(f"    Total score: {scores['total']:.2%}")
        
        return scores
    
    def evaluate_model(self, model_config: ModelConfig, 
                       exams: List[ExamConfig],
                       skip_existing: bool = True) -> Optional[Dict[str, Dict[str, float]]]:
        """Evaluate a single model on all exams."""
        
        # Create model output directory path
        model_name_safe = model_config.name.replace("/", "_").replace(":", "_")
        model_output_dir = self.output_base_dir / model_name_safe
        
        # Check if already evaluated
        summary_file = model_output_dir / "summary.json"
        if skip_existing and summary_file.exists():
            self.logger.info(f"Skipping {model_config.name} - already evaluated (found {summary_file})")
            # Load and return existing results
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"Evaluating model: {model_config.name}")
        self.logger.info(f"=" * 60)
        
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model_wrapper = UnslothModelWrapper(model_config, self.logger)
        model_wrapper.load()
        
        # Save model configuration (after loading so we can include chat template info)
        config_file = model_output_dir / "model_config.json"
        chat_template_used = None
        if hasattr(model_wrapper.tokenizer, 'chat_template') and model_wrapper.tokenizer.chat_template:
            chat_template_used = model_wrapper.tokenizer.chat_template
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({
                "name": model_config.name,
                "base_model": model_config.base_model,
                "lora_path": model_config.lora_path,
                "max_seq_length": model_config.max_seq_length,
                "load_in_4bit": model_config.load_in_4bit,
                "chat_template_override": model_config.chat_template,
                "chat_template_used": chat_template_used,
            }, f, indent=2, ensure_ascii=False)
        
        # Run all exams
        all_scores = {}
        for exam in exams:
            scores = self.evaluate_model_on_exam(model_wrapper, exam, model_output_dir)
            all_scores[exam.level] = scores
        
        # Unload model
        model_wrapper.unload()
        
        # Save summary for this model
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2)
        
        return all_scores
    
    def evaluate_all(self, models: List[ModelConfig], 
                     exams: List[ExamConfig],
                     skip_existing: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate all models on all exams.
        
        Args:
            models: List of model configurations to evaluate
            exams: List of exam configurations to run
            skip_existing: If True, skip models that already have results (default: True)
        """
        self.logger.info("Starting Luxembourgish Language Exam Evaluation")
        self.logger.info(f"Models to evaluate: {len(models)}")
        self.logger.info(f"Exams per model: {len(exams)}")
        self.logger.info(f"Skip existing: {skip_existing}")
        
        all_results = {}
        
        for model_config in models:
            try:
                scores = self.evaluate_model(model_config, exams, skip_existing=skip_existing)
                all_results[model_config.name] = scores
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_config.name}: {e}")
                all_results[model_config.name] = {"error": str(e)}
        
        # Generate final summary
        self.generate_summary(all_results, exams)
        
        return all_results
    
    def generate_summary(self, all_results: Dict[str, Dict[str, Dict[str, float]]],
                         exams: List[ExamConfig]):
        """Generate a summary TSV file with all results."""
        self.logger.info("Generating final summary")
        
        # Get all unique categories across exams
        all_categories = set()
        for model_results in all_results.values():
            for exam_results in model_results.values():
                if isinstance(exam_results, dict) and "error" not in exam_results:
                    all_categories.update(exam_results.keys())
        
        all_categories = sorted(all_categories)
        
        # Create summary file
        summary_file = self.output_base_dir / "final_summary.tsv"
        with open(summary_file, "w", encoding="utf-8") as f:
            # Header
            headers = ["Model"]
            for exam in exams:
                for cat in all_categories:
                    headers.append(f"{exam.level.upper()}_{cat}")
            f.write("\t".join(headers) + "\n")
            
            # Data rows
            for model_name, exam_results in all_results.items():
                row = [model_name]
                for exam in exams:
                    exam_scores = exam_results.get(exam.level, {})
                    for cat in all_categories:
                        if isinstance(exam_scores, dict) and "error" not in exam_scores:
                            score = exam_scores.get(cat, "N/A")
                            if isinstance(score, float):
                                row.append(f"{score:.4f}")
                            else:
                                row.append(str(score))
                        else:
                            row.append("ERROR")
                f.write("\t".join(row) + "\n")
        
        self.logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main entry point for the evaluation script."""
    import argparse
    
    print("\n" + "=" * 60)
    print("LUXEMBOURGISH LANGUAGE EXAM EVALUATION")
    print("=" * 60 + "\n")
    
    parser = argparse.ArgumentParser(description="Evaluate models on Luxembourgish language exams")
    parser.add_argument("--force", "-f", action="store_true", 
                        help="Force re-evaluation of all models (ignore existing results)")
    args = parser.parse_args()
    
    # Import configuration from config.py
    # Edit config.py to add/remove models and exams
    from config import MODELS, EXAMS, OUTPUT_DIR
    
    print(f"Output directory: {OUTPUT_DIR}") 
    print(f"Models to evaluate: {len(MODELS)}")
    print(f"Exam levels: {[e.level.upper() for e in EXAMS]}")
    print(f"Skip existing: {not args.force}")
    print()
    
    skip_existing = not args.force
    
    evaluator = LuxembourgishEvaluator(output_base_dir=OUTPUT_DIR)
    results = evaluator.evaluate_all(MODELS, EXAMS, skip_existing=skip_existing)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()