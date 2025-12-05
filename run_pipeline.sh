#!/usr/bin/env bash
#===============================================================================
# Silicon Philosophers Pipeline Runner
#===============================================================================
# A bash script to run the complete pipeline:
# - Data crawling (PhilPeople profiles, PhilPapers survey)
# - LLM evaluation
# - DPO training
# - Analysis
#
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-crawl        Skip crawling step (use existing data)
#   --skip-eval         Skip model evaluation step
#   --skip-dpo          Skip DPO training step
#   --skip-analysis     Skip analysis step
#   --limit N           Limit crawling to N profiles
#   --model MODEL       Model to use (default: Qwen/Qwen2.5-0.5B-Instruct)
#   --data-dir DIR      Directory containing data files (default: parent dir)
#   --help              Show this help message
#
# Examples:
#   ./run_pipeline.sh                           # Run full pipeline
#   ./run_pipeline.sh --skip-crawl              # Use existing crawled data
#   ./run_pipeline.sh --limit 10                # Crawl only 10 profiles
#   ./run_pipeline.sh --skip-crawl --skip-dpo   # Only eval and analysis
#===============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model to use
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Crawl limit (0 = all)
CRAWL_LIMIT=0

# Skip flags
SKIP_CRAWL=false
SKIP_EVAL=false
SKIP_DPO=false
SKIP_ANALYSIS=false

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR=""  # Will default to parent of SCRIPT_DIR if not set
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "========================================================================"
    echo " $1"
    echo "========================================================================"
}

show_help() {
    head -30 "$0" | tail -25
    exit 0
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-crawl)
            SKIP_CRAWL=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-dpo)
            SKIP_DPO=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --limit)
            CRAWL_LIMIT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Set default data directory to parent of script dir (project root)
if [[ -z "$DATA_DIR" ]]; then
    DATA_DIR="$(dirname "$SCRIPT_DIR")"
fi

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print_header "SILICON PHILOSOPHERS PIPELINE"

echo ""
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Crawl Limit:     ${CRAWL_LIMIT:-all}"
echo "  Data Directory:  $DATA_DIR"
echo "  Script Dir:      $SCRIPT_DIR"
echo ""
echo "Skip Flags:"
echo "  Skip Crawl:      $SKIP_CRAWL"
echo "  Skip Eval:       $SKIP_EVAL"
echo "  Skip DPO:        $SKIP_DPO"
echo "  Skip Analysis:   $SKIP_ANALYSIS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to data directory (where JSON files are)
cd "$DATA_DIR"
log_info "Working directory: $(pwd)"

# ============================================================================
# STEP 1: DATA CRAWLING
# ============================================================================

if [[ "$SKIP_CRAWL" == "false" ]]; then
    print_header "STEP 1: DATA CRAWLING"
    
    log_info "Crawling PhilPeople profiles..."
    
    if [[ $CRAWL_LIMIT -gt 0 ]]; then
        python3 "$SCRIPT_DIR/1a_crawl_philpeople_views.py" --limit "$CRAWL_LIMIT"
    else
        python3 "$SCRIPT_DIR/1a_crawl_philpeople_views.py"
    fi
    
    log_info "Scraping profile details..."
    python3 "$SCRIPT_DIR/1b_scrape_profile_details.py"
    
    log_info "Crawling PhilPapers survey (may be blocked by Cloudflare)..."
    if [[ $CRAWL_LIMIT -gt 0 ]]; then
        python3 "$SCRIPT_DIR/1c_crawl_philpapers_survey.py" --limit "$CRAWL_LIMIT" || {
            log_warning "PhilPapers crawl blocked. Use 1d_parse_manual_download.py for manual downloads."
        }
    else
        python3 "$SCRIPT_DIR/1c_crawl_philpapers_survey.py" || {
            log_warning "PhilPapers crawl blocked. Use 1d_parse_manual_download.py for manual downloads."
        }
    fi
    
    log_success "Data crawling complete"
else
    log_info "Skipping crawl step (using existing data)"
fi

# ============================================================================
# STEP 2: DATA PROCESSING
# ============================================================================

print_header "STEP 2: DATA PROCESSING"

log_info "Processing crawled HTML..."
python3 "$SCRIPT_DIR/2a_process_crawled_html.py" || log_warning "No HTML files to process"

log_info "Merging philosopher data..."
python3 "$SCRIPT_DIR/2b_merge_philosopher_data.py" || log_warning "Merge step skipped"

log_info "Merging survey with philosophers..."
python3 "$SCRIPT_DIR/2c_merge_survey_with_philosophers.py" || log_warning "Survey merge skipped"

log_success "Data processing complete"

# ============================================================================
# STEP 3: LLM EVALUATION
# ============================================================================

if [[ "$SKIP_EVAL" == "false" ]]; then
    print_header "STEP 3: LLM EVALUATION"
    
    log_info "Running model evaluation with $MODEL..."
    python3 "$SCRIPT_DIR/3_model_eval.py"
    
    log_success "Model evaluation complete"
else
    log_info "Skipping evaluation step"
fi

# ============================================================================
# STEP 4: POST-PROCESSING
# ============================================================================

print_header "STEP 4: POST-PROCESSING"

log_info "Processing model results..."
python3 "$SCRIPT_DIR/4a_process_model_results.py" || log_warning "No results to process"

log_info "Merging LLM responses..."
python3 "$SCRIPT_DIR/4b_merge_llm_responses.py" || log_warning "No responses to merge"

log_info "Normalizing question names..."
python3 "$SCRIPT_DIR/4c_normalize_question_names.py" || log_warning "Normalization skipped"

log_success "Post-processing complete"

# ============================================================================
# STEP 5: ANALYSIS
# ============================================================================

if [[ "$SKIP_ANALYSIS" == "false" ]]; then
    print_header "STEP 5: ANALYSIS"
    
    log_info "Computing quality metrics..."
    python3 "$SCRIPT_DIR/5a_compute_quality_metrics.py" || log_warning "Quality metrics skipped"
    
    log_info "Running correlation analysis..."
    python3 "$SCRIPT_DIR/5b_correlation_analysis.py" || log_warning "Correlation analysis skipped"
    
    log_info "Computing all model correlations..."
    python3 "$SCRIPT_DIR/5c_compute_all_model_correlations.py" || log_warning "Model correlations skipped"
    
    log_success "Analysis complete"
else
    log_info "Skipping analysis step"
fi

# ============================================================================
# STEP 6-7: DPO TRAINING
# ============================================================================

if [[ "$SKIP_DPO" == "false" ]]; then
    print_header "STEP 6-7: DPO TRAINING"
    
    log_info "Preparing DPO dataset (anonymized)..."
    python3 "$SCRIPT_DIR/6_prepare_dpo_dataset.py"
    
    log_info "Fine-tuning with DPO..."
    python3 "$SCRIPT_DIR/7_finetune_dpo.py"
    
    log_success "DPO training complete"
    
    # ============================================================================
    # STEP 7b: EVALUATE FINE-TUNED MODEL
    # ============================================================================
    
    print_header "STEP 7b: EVALUATE FINE-TUNED MODEL"
    
    log_info "Evaluating fine-tuned model..."
    python3 "$SCRIPT_DIR/3_model_eval.py" --finetuned --provider huggingface
    
    log_info "Processing fine-tuned model results..."
    python3 "$SCRIPT_DIR/4a_process_model_results.py" || log_warning "No results to process"
    python3 "$SCRIPT_DIR/4b_merge_llm_responses.py" || log_warning "No responses to merge"
    python3 "$SCRIPT_DIR/4c_normalize_question_names.py" || log_warning "Normalization skipped"
    
    log_info "Computing quality metrics for fine-tuned model..."
    python3 "$SCRIPT_DIR/5a_compute_quality_metrics.py" || log_warning "Quality metrics skipped"
    
    log_success "Fine-tuned model evaluation complete"
else
    log_info "Skipping DPO training step"
fi

# ============================================================================
# STEP 8: COMPARISON
# ============================================================================

if [[ "$SKIP_DPO" == "false" ]]; then
    print_header "STEP 8: COMPARISON"
    
    log_info "Comparing base vs fine-tuned model..."
    python3 "$SCRIPT_DIR/8_compare_finetuning.py"
    
    log_success "Comparison complete"
fi

# ============================================================================
# SUMMARY
# ============================================================================

print_header "PIPELINE COMPLETE"

echo ""
log_success "All pipeline steps completed successfully!"
echo ""
echo "Output files:"
echo "  - Crawled data: views_html/, philosopher_profiles/"
echo "  - Processed data: philosophers_with_countries.json"
echo "  - Model responses: llm_responses_*/"
echo "  - Quality metrics: model_correlations/"
echo "  - DPO model: qwen2.5_0.5b_philosopher_dpo/"
echo "  - Fine-tuned responses: llm_responses_huggingface/ (with --finetuned)"
echo "  - Comparison: finetuning_comparison.txt"
echo ""
