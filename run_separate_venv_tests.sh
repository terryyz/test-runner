#!/bin/bash
# Script to run tests with separate virtual environments for each repository

# Make sure this script is executable
# chmod +x run_separate_venv_tests.sh

echo "Running tests with separate virtual environments for each repository..."

# Check if input file exists
if [ -f "$1" ]; then
    INPUT_FILE="$1"
else
    INPUT_FILE="test.jsonl"
    echo "Using default input file: $INPUT_FILE"
fi

# Generate output filename based on input
OUTPUT_FILE="${INPUT_FILE%.*}_results.jsonl"

# Run the tests with separate virtual environments
python run_tests.py --input-file "$INPUT_FILE" --output-file "$OUTPUT_FILE" --use-separate-envs --verbose

# Check exit status
if [ $? -eq 0 ]; then
    echo "Tests completed successfully. Results saved to $OUTPUT_FILE"
else
    echo "Tests failed. Check logs for more information."
fi 