#!/bin/bash
# Example script to demonstrate how to use test-runner

# Create a directory for output
mkdir -p output

# Copy the example test.jsonl file to the output directory
cp ../example_test.jsonl output/test.jsonl

# Run the test runner with verbose output
python ../run_tests.py --output-dir output --verbose --num-workers 2

# View the results
echo "Results written to output/test_results.jsonl:"
cat output/test_results.jsonl 