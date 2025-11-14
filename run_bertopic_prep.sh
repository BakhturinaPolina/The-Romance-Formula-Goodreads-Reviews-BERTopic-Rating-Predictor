#!/bin/bash
# Wrapper script to run prepare_bertopic_input.py with proper logging and venv
# This is a convenience wrapper - main script is in src/reviews_analysis/bertopic_preparation/

PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research"
cd "$PROJECT_ROOT" || exit 1

# Call the actual script from the organized location
exec "$PROJECT_ROOT/src/reviews_analysis/bertopic_preparation/run_bertopic_prep.sh"
