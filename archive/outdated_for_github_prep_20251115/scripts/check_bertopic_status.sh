#!/bin/bash
# Quick status check for BERTopic preparation
# This is a convenience wrapper - main script is in src/reviews_analysis/bertopic_preparation/

PROJECT_ROOT="/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research"
cd "$PROJECT_ROOT" || exit 1

# Call the actual script from the organized location
exec "$PROJECT_ROOT/src/reviews_analysis/bertopic_preparation/check_bertopic_status.sh"
