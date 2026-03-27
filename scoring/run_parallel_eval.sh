#!/bin/bash
# =====================================================
# Parallel Luxembourgish Dataset Evaluation Runner
# =====================================================
# This script sets up and runs parallel evaluation using tmux.
#
# Usage:
#   ./run_parallel_eval.sh <dataset_path> <output_base> <num_workers> [options]
#
# Examples:
#   ./run_parallel_eval.sh data/dataset.jsonl results/evaluated 4
#   ./run_parallel_eval.sh data/dataset.jsonl results/evaluated 8 --model gpt-5-mini
#
# To monitor: tmux attach-session -t luxeval
# To switch windows: Ctrl+B then window number (0-N)
# =====================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL="gpt-5-mini"
RATE_LIMIT_DELAY="0.5"
SESSION_NAME="luxeval"
PYTHON_SCRIPT="lux_dataset_evaluator.py"
MICROMAMBA_ENV="unsloth_env"

# Parse arguments
if [ $# -lt 3 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 <dataset_path> <output_base> <num_workers> [options]"
    echo ""
    echo "Required arguments:"
    echo "  dataset_path   Path to the input dataset (JSONL, CSV, or JSON)"
    echo "  output_base    Base path for output files (without extension)"
    echo "  num_workers    Number of parallel workers"
    echo ""
    echo "Optional arguments:"
    echo "  --model MODEL              OpenAI model (default: gpt-5-mini)"
    echo "  --rate-limit-delay DELAY   Delay between API calls (default: 0.3)"
    echo "  --session-name NAME        Tmux session name (default: luxeval)"
    echo "  --force                    Force re-evaluation"
    echo ""
    echo "Examples:"
    echo "  $0 data/dataset.jsonl results/evaluated 4"
    echo "  $0 data/dataset.jsonl results/evaluated 8 --model gpt-5-mini"
    exit 1
fi

DATASET_PATH="$1"
OUTPUT_BASE="$2"
NUM_WORKERS="$3"
shift 3

FORCE_FLAG=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --rate-limit-delay)
            RATE_LIMIT_DELAY="$2"
            shift 2
            ;;
        --session-name)
            SESSION_NAME="$2"
            shift 2
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset file not found: $DATASET_PATH${NC}"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Check for OpenAI API key - first try to load from .env file
if [ -f ".env" ]; then
    echo -e "${YELLOW}Loading environment from .env file...${NC}"
    # Source .env file (handles KEY=value format)
    set -a
    source .env
    set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not found${NC}"
    echo ""
    echo "Please either:"
    echo "  1. Create a .env file with: OPENAI_API_KEY=your-api-key"
    echo "  2. Or export it: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_BASE")
if [ ! -d "$OUTPUT_DIR" ] && [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"
fi

# Create directories for logs and checkpoints
mkdir -p logs checkpoints

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Tmux session '$SESSION_NAME' already exists.${NC}"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach-session -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    echo ""
    read -p "Do you want to kill the existing session and start fresh? (y/N): " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo -e "${GREEN}Killed existing session${NC}"
    else
        echo "Exiting. Attach to existing session with: tmux attach-session -t $SESSION_NAME"
        exit 0
    fi
fi

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Parallel Evaluation Configuration${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "Dataset:        ${GREEN}$DATASET_PATH${NC}"
echo -e "Output base:    ${GREEN}$OUTPUT_BASE${NC}"
echo -e "Workers:        ${GREEN}$NUM_WORKERS${NC}"
echo -e "Model:          ${GREEN}$MODEL${NC}"
echo -e "Rate limit:     ${GREEN}${RATE_LIMIT_DELAY}s${NC}"
echo -e "Session name:   ${GREEN}$SESSION_NAME${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Count dataset entries
echo -e "${YELLOW}Counting dataset entries...${NC}"
if [[ "$DATASET_PATH" == *.jsonl ]]; then
    TOTAL_ENTRIES=$(wc -l < "$DATASET_PATH")
elif [[ "$DATASET_PATH" == *.csv ]]; then
    TOTAL_ENTRIES=$(($(wc -l < "$DATASET_PATH") - 1))  # Subtract header
else
    TOTAL_ENTRIES="unknown"
fi
echo -e "Total entries: ${GREEN}$TOTAL_ENTRIES${NC}"
if [ "$TOTAL_ENTRIES" != "unknown" ]; then
    ENTRIES_PER_WORKER=$((TOTAL_ENTRIES / NUM_WORKERS))
    echo -e "Entries per worker: ~${GREEN}$ENTRIES_PER_WORKER${NC}"
fi
echo ""

# Create tmux session with first window
echo -e "${YELLOW}Creating tmux session '$SESSION_NAME'...${NC}"
tmux new-session -d -s "$SESSION_NAME" -n "worker0"

# Start workers
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_OUTPUT="${OUTPUT_BASE}_worker${i}.jsonl"
    
    CMD="python3 $PYTHON_SCRIPT \
        --dataset $DATASET_PATH \
        --output $WORKER_OUTPUT \
        --num-workers $NUM_WORKERS \
        --worker-id $i \
        --model $MODEL \
        --rate-limit-delay $RATE_LIMIT_DELAY \
        $FORCE_FLAG"
    
    if [ $i -eq 0 ]; then
        # First worker uses the existing window
        tmux send-keys -t "$SESSION_NAME:worker0" "micromamba activate $MICROMAMBA_ENV" Enter
        tmux send-keys -t "$SESSION_NAME:worker0" "$CMD" Enter
    else
        # Create new window for subsequent workers
        tmux new-window -t "$SESSION_NAME" -n "worker$i"
        tmux send-keys -t "$SESSION_NAME:worker$i" "micromamba activate $MICROMAMBA_ENV" Enter
        tmux send-keys -t "$SESSION_NAME:worker$i" "$CMD" Enter
    fi
    
    echo -e "  ${GREEN}✓${NC} Started worker $i -> $WORKER_OUTPUT"
done

# Create a monitoring window
tmux new-window -t "$SESSION_NAME" -n "monitor"
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Monitoring Commands:'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=================='" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Watch checkpoint progress:'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '  watch -n 5 \"ls -la checkpoints/\"'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Watch output files:'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '  watch -n 10 \"wc -l ${OUTPUT_BASE}_worker*.jsonl 2>/dev/null || echo No output yet\"'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'View latest logs:'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '  tail -f logs/evaluation_worker*_*.log'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Merge results (after all complete):'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo '  python3 $PYTHON_SCRIPT --merge --output $OUTPUT_BASE --num-workers $NUM_WORKERS'" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:monitor" "echo 'Switch windows: Ctrl+B then 0-$((NUM_WORKERS))'" Enter

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}All workers started successfully!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo -e "To attach to the tmux session:"
echo -e "  ${BLUE}tmux attach-session -t $SESSION_NAME${NC}"
echo ""
echo -e "Tmux navigation:"
echo -e "  ${YELLOW}Ctrl+B, 0-$((NUM_WORKERS-1))${NC}  - Switch to worker window"
echo -e "  ${YELLOW}Ctrl+B, $NUM_WORKERS${NC}         - Switch to monitor window"
echo -e "  ${YELLOW}Ctrl+B, n${NC}           - Next window"
echo -e "  ${YELLOW}Ctrl+B, p${NC}           - Previous window"
echo -e "  ${YELLOW}Ctrl+B, d${NC}           - Detach (processes keep running)"
echo ""
echo -e "After all workers complete, merge results:"
echo -e "  ${BLUE}python3 $PYTHON_SCRIPT --merge --output $OUTPUT_BASE --num-workers $NUM_WORKERS${NC}"
echo ""

# Ask if user wants to attach
read -p "Attach to tmux session now? (Y/n): " attach
if [[ $attach != [nN] && $attach != [nN][oO] ]]; then
    tmux attach-session -t "$SESSION_NAME"
fi