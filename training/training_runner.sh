#!/bin/bash

# Universal Fine-tuning Runner Script

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="config.yaml"
CHECKPOINT_DIR=""
USE_WANDB=""
MAX_STEPS=""
EVAL_ONLY=""
INSTALL_DEPS=false
BACKUP=false
MONITOR=false
MODEL_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""  # No default - respect YAML

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Function to extract value from YAML
get_yaml_value() {
    local file=$1
    local key=$2
    local section=$3
    
    if [ -f "$file" ]; then
        # Try to extract value from YAML (basic extraction)
        if [ -n "$section" ]; then
            # Look for value under a section
            awk "/^$section:/{flag=1} flag && /$key:/{print \$2; exit}" "$file" | tr -d '"' | tr -d "'"
        else
            # Look for value at root level
            grep "^$key:" "$file" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'"
        fi
    fi
}

# Function to check if required dependencies are installed
check_dependencies() {
    print_message "Checking dependencies..."
    
    local all_found=true
    
    python -c "import torch" 2>/dev/null || {
        print_error "PyTorch not found"
        all_found=false
    }
    
    python -c "import unsloth" 2>/dev/null || {
        print_error "Unsloth not found"
        all_found=false
    }
    
    python -c "import trl" 2>/dev/null || {
        print_error "TRL not found"
        all_found=false
    }
    
    python -c "import yaml" 2>/dev/null || {
        print_warning "PyYAML not found (needed for config reading)"
    }
    
    if $all_found; then
        print_message "All required dependencies found!"
        return 0
    else
        return 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_message "Installing dependencies..."
    
    # Install Unsloth
    #pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install unsloth unsloth-zoo
    
    # Install other requirements
    pip install "transformers==4.56.2" "trl==0.22.2" peft accelerate bitsandbytes
    pip install evaluate rouge_score pandas datasets
    pip install pyyaml tqdm
    
    # Optional: Install Weights & Biases
    read -p "Install Weights & Biases for experiment tracking? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install wandb
    fi
    
    print_message "Dependencies installed successfully!"
}

# Function to create directory structure
setup_directories() {
    local output_dir=$1
    print_message "Setting up directory structure in $output_dir..."
    
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/logs"
    mkdir -p "$output_dir/checkpoints"
    mkdir -p "$output_dir/configs"
    mkdir -p "$output_dir/dataset_cache"
    
    print_message "Directories created!"
}

# Function to backup existing output
backup_existing_output() {
    local output_dir=$1
    if [ -d "$output_dir" ] && [ "$(ls -A $output_dir)" ]; then
        local backup_dir="${output_dir}_backup_$(date +%Y%m%d_%H%M%S)"
        print_warning "Existing output found. Backing up to $backup_dir"
        mv "$output_dir" "$backup_dir"
    fi
}

# Function to display configuration summary
show_config_summary() {
    echo ""
    echo "========================================="
    echo "   Configuration Summary"
    echo "========================================="
    echo "Config file:     $CONFIG_FILE"
    
    # Read values from YAML if possible
    if [ -f "$CONFIG_FILE" ]; then
        local yaml_output=$(get_yaml_value "$CONFIG_FILE" "output_dir" "training")
        local yaml_model=$(get_yaml_value "$CONFIG_FILE" "model_name" "model")
        local yaml_dataset=$(get_yaml_value "$CONFIG_FILE" "dataset_name" "training")
        
        [ -n "$yaml_model" ] && echo "Model:           $yaml_model"
        [ -n "$yaml_dataset" ] && echo "Dataset:         $yaml_dataset"
        
        # Show effective output directory
        if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
            echo "Output dir:      $OUTPUT_DIR_OVERRIDE (override)"
        elif [ -n "$yaml_output" ]; then
            echo "Output dir:      $yaml_output (from config)"
        else
            echo "Output dir:      ./output (default)"
        fi
    else
        echo "Output dir:      ${OUTPUT_DIR_OVERRIDE:-from config}"
    fi
    
    [ -n "$CHECKPOINT_DIR" ] && echo "Resume from:     $CHECKPOINT_DIR"
    [ -n "$MODEL_OVERRIDE" ] && echo "Model override:  $MODEL_OVERRIDE"
    [ -n "$MAX_STEPS" ] && echo "Max steps:       $MAX_STEPS"
    [ "$USE_WANDB" = "true" ] && echo "W&B enabled:     Yes"
    [ "$EVAL_ONLY" = "true" ] && echo "Mode:            Evaluation only"
    [ "$MONITOR" = "true" ] && echo "Monitoring:      Enabled"
    echo "========================================="
    echo ""
}

# Function to monitor training
monitor_training() {
    local output_dir=$1
    local log_pattern="$output_dir/logs/training_*.log"
    
    print_message "Monitoring training logs..."
    print_message "Press Ctrl+C to stop monitoring (training will continue)"
    
    # Wait for log file to be created
    local wait_count=0
    while [ ! -f $log_pattern ] && [ $wait_count -lt 30 ]; do
        sleep 1
        ((wait_count++))
    done
    
    if [ $wait_count -eq 30 ]; then
        print_warning "Log file not found after 30 seconds"
        print_message "Check if training started properly"
        return 1
    fi
    
    # Tail the most recent log file
    local latest_log=$(ls -t $log_pattern 2>/dev/null | head -1)
    if [ -f "$latest_log" ]; then
        tail -f "$latest_log"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --model)
            MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --resume)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="true"
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --eval-only)
            EVAL_ONLY="true"
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --backup)
            BACKUP=true
            shift
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE        Configuration file (default: config.yaml)"
            echo "  --output-dir DIR     Override output directory from config"
            echo "  --model NAME         Override model name from config"
            echo "  --resume DIR         Resume from checkpoint directory"
            echo "  --wandb              Enable Weights & Biases logging"
            echo "  --max-steps N        Override maximum training steps"
            echo "  --eval-only          Only run evaluation"
            echo "  --install-deps       Install required dependencies"
            echo "  --backup             Backup existing output before training"
            echo "  --monitor            Monitor training logs in real-time"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Standard training with config"
            echo "  $0 --config config.yaml"
            echo ""
            echo "  # Resume training with monitoring"
            echo "  $0 --config config.yaml --resume ./checkpoints/checkpoint-500 --monitor"
            echo ""
            echo "  # Override model and output"
            echo "  $0 --config config.yaml --model unsloth/gemma-3-1b-it --output-dir ./output_gemma"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Header
echo "========================================="
echo "   Universal Fine-tuning Runner v2.0"
echo "========================================="
echo ""

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    install_dependencies
    exit 0
fi

# Check dependencies
if ! check_dependencies; then
    print_error "Missing dependencies!"
    print_message "Run with --install-deps to install required packages"
    print_message "Or activate your conda environment if using one"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    print_message "Please create a configuration file or specify one with --config"
    exit 1
fi

# Detect output directory from config if not overridden
if [ -z "$OUTPUT_DIR_OVERRIDE" ]; then
    OUTPUT_DIR=$(get_yaml_value "$CONFIG_FILE" "output_dir" "training")
    OUTPUT_DIR=${OUTPUT_DIR:-"./output"}  # Default if not in YAML
else
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
fi

# Backup if requested
if [ "$BACKUP" = true ]; then
    backup_existing_output "$OUTPUT_DIR"
fi

# Setup directories
setup_directories "$OUTPUT_DIR"

# Show configuration summary
show_config_summary

# Build command - Only add arguments that were explicitly set
CMD="python multi_model_finetuning.py"

# Always add config
CMD="$CMD --config $CONFIG_FILE"

# Only add overrides if explicitly specified
if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR_OVERRIDE"
    print_debug "Overriding output directory: $OUTPUT_DIR_OVERRIDE"
fi

if [ -n "$MODEL_OVERRIDE" ]; then
    CMD="$CMD --model $MODEL_OVERRIDE"
    print_debug "Overriding model: $MODEL_OVERRIDE"
fi

if [ -n "$CHECKPOINT_DIR" ]; then
    CMD="$CMD --resume $CHECKPOINT_DIR"
    print_message "Resuming from checkpoint: $CHECKPOINT_DIR"
fi

if [ "$USE_WANDB" = "true" ]; then
    CMD="$CMD --wandb"
    print_message "Weights & Biases logging enabled"
    
    # Check if wandb is logged in
    if ! wandb status 2>/dev/null | grep -q "Logged in"; then
        print_warning "You may need to login to Weights & Biases"
        print_message "Run: wandb login"
    fi
fi

if [ -n "$MAX_STEPS" ]; then
    CMD="$CMD --max-steps $MAX_STEPS"
    print_message "Max steps override: $MAX_STEPS"
fi

if [ "$EVAL_ONLY" = "true" ]; then
    CMD="$CMD --eval-only"
    print_message "Running evaluation only"
fi

# Display the command that will be run
echo ""
print_message "Command to execute:"
echo "  $CMD"
echo ""

# Confirm before starting (skip for eval-only)
if [ "$EVAL_ONLY" != "true" ]; then
    read -p "Start training? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_message "Training cancelled"
        exit 0
    fi
fi

# Run training
print_message "Starting process..."
echo ""

# Start training in background if monitoring is requested
if [ "$MONITOR" = true ]; then
    # Create a unique log file for console output
    CONSOLE_LOG="$OUTPUT_DIR/console_output_$(date +%Y%m%d_%H%M%S).log"
    
    # Start training in background
    $CMD > "$CONSOLE_LOG" 2>&1 &
    TRAINING_PID=$!
    
    print_message "Training started in background (PID: $TRAINING_PID)"
    print_message "Console output saved to: $CONSOLE_LOG"
    
    # Give it a moment to start and create log files
    sleep 3
    
    # Check if process is still running
    if ! ps -p $TRAINING_PID > /dev/null; then
        print_error "Training process failed to start!"
        print_message "Check console output: $CONSOLE_LOG"
        tail -20 "$CONSOLE_LOG"
        exit 1
    fi
    
    # Monitor logs
    monitor_training "$OUTPUT_DIR"
    
    # After monitoring stops, remind user that training continues
    print_message "Monitoring stopped. Training continues in background (PID: $TRAINING_PID)"
    print_message "To check if still running: ps -p $TRAINING_PID"
    print_message "To view logs: tail -f $OUTPUT_DIR/logs/training_*.log"
else
    # Run training in foreground
    $CMD
    EXIT_CODE=$?
fi

# Check exit status (for foreground execution)
if [ "$MONITOR" != "true" ]; then
    if [ $EXIT_CODE -eq 0 ]; then
        print_message "Process completed successfully!"
        print_message "Results saved to: $OUTPUT_DIR"
        
        # Show summary of results if available
        if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
            echo ""
            echo "Training Summary:"
            echo "-----------------"
            python -c "import json; data=json.load(open('$OUTPUT_DIR/training_summary.json')); print(json.dumps(data, indent=2))" 2>/dev/null || cat "$OUTPUT_DIR/training_summary.json"
        fi
        
        # Show evaluation results if available
        if [ -f "$OUTPUT_DIR/evaluation_results.json" ]; then
            echo ""
            echo "Evaluation Results:"
            echo "-------------------"
            python -c "import json; data=json.load(open('$OUTPUT_DIR/evaluation_results.json')); print(json.dumps(data, indent=2))" 2>/dev/null || cat "$OUTPUT_DIR/evaluation_results.json"
        fi
    else
        print_error "Process failed! Check logs for details"
        print_message "Logs location: $OUTPUT_DIR/logs/"
        
        # Show last few lines of error log if it exists
        ERROR_LOG=$(ls -t $OUTPUT_DIR/logs/errors_*.log 2>/dev/null | head -1)
        if [ -f "$ERROR_LOG" ]; then
            echo ""
            echo "Last errors:"
            echo "------------"
            tail -10 "$ERROR_LOG"
        fi
        exit $EXIT_CODE
    fi
fi

echo ""
echo "========================================="
echo "   Process Complete"
echo "========================================="