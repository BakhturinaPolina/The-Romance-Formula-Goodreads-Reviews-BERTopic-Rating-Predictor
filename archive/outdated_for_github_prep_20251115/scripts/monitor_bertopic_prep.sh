#!/bin/bash
# Monitoring script for prepare_bertopic_input.py
# Monitors progress and displays updates
# This is a convenience wrapper - main script is in src/reviews_analysis/bertopic_preparation/

PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research"
cd "$PROJECT_ROOT" || exit 1

# Call the actual script from the organized location
exec "$PROJECT_ROOT/src/reviews_analysis/bertopic_preparation/monitor_bertopic_prep.sh"
