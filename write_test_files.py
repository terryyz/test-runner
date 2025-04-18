#!/usr/bin/env python3

import argparse
import json
import os
import sys
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm module not found. Install it using 'pip install tqdm' for progress bars.")
    # Create a simple tqdm replacement that does nothing
    class FakeTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        
        def __iter__(self):
            return iter(self.iterable)
        
        def write(self, s, **kwargs):
            print(s)
    
    tqdm = FakeTqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Write test files to repository paths based on input JSONL")
    parser.add_argument("input_file", help="Input JSONL file containing repository and test file information")
    parser.add_argument("--working-dir", "-w", dest="working_dir", help="Base working directory for repositories")
    parser.add_argument("--skip-pattern", "-s", dest="skip_pattern", 
                        help="Skip repositories matching this regex pattern")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry run (don't actually write files)")
    parser.add_argument("--convert-format", "-c", dest="convert_output", 
                        help="Generate test metadata in format compatible with run_tests.py and write to specified file")
    return parser.parse_args()


def generate_test_filename(original_file, index):
    """Generate a test filename based on the original file name and index."""
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    return f"test_{base_name}_{index}.py"


def process_jsonl(input_file, working_dir=None, skip_pattern=None, dry_run=False, convert_output=None):
    """Process the input JSONL file and write test files."""
    processed_repos = 0
    skipped_repos = 0
    written_files = 0
    skipped_files = 0
    
    # Structure to store test metadata for conversion format
    test_metadata = {}

    # Compile regex pattern if provided
    skip_regex = None
    if skip_pattern:
        try:
            skip_regex = re.compile(skip_pattern)
            print(f"Will skip repositories matching pattern: {skip_pattern}")
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{skip_pattern}', ignoring. Error: {str(e)}")
    
    print(f"Processing repositories from {input_file}")
    
    # Count total lines first for the progress bar
    total_lines = sum(1 for _ in open(input_file, 'r'))
    
    # Read the JSONL file line by line with progress bar
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Processing repositories", unit="repo"), 1):
            try:
                # Parse the JSON object from each line
                repo_data = json.loads(line)
                repo_path = repo_data.get("repository")
                
                if not repo_path:
                    tqdm.write(f"Warning: Line {line_num} - Missing repository path, skipping")
                    continue
                
                # Check if repository matches skip pattern
                if skip_regex and skip_regex.search(repo_path):
                    tqdm.write(f"Skipping repository (matches skip pattern): {repo_path}")
                    skipped_repos += 1
                    continue
                
                # Initialize test metadata for this repository
                if convert_output:
                    test_metadata[repo_path] = {
                        "repository": repo_path,
                        "tests": []
                    }
                
                # Adjust repository path if working_dir is provided
                adjusted_repo_path = repo_path
                if working_dir:
                    adjusted_repo_path = os.path.join(working_dir, repo_path)
                
                # Check if repository directory exists
                if not os.path.isdir(adjusted_repo_path):
                    tqdm.write(f"Warning: Repository directory not found: {adjusted_repo_path}")
                    # Count total test contents across all test files
                    test_count = sum(len(test_info.get("content", [])) 
                                    for test_info in repo_data.get("tests", {}).values() 
                                    if isinstance(test_info.get("content", []), list))
                    tqdm.write(f"  Skipped {test_count} test files")
                    skipped_files += test_count
                    continue
                
                tqdm.write(f"\nProcessing repository: {adjusted_repo_path}")
                processed_repos += 1
                
                # Process each tested file from the "tests" dictionary
                test_files = repo_data.get("tests", {})
                for tested_file, test_info in tqdm(test_files.items(), desc="  Processing tested files", unit="file", leave=False):
                    path = test_info.get("path")
                    content_list = test_info.get("content", [])
                    dependencies = test_info.get("dependencies", [])
                    
                    # Skip if content is empty
                    if not content_list:
                        tqdm.write(f"  Warning: No test content for {tested_file}, skipping")
                        skipped_files += 1
                        continue
                    
                    # Use the path of the tested file to determine where to write tests
                    if not path:
                        # If path is not specified, use the tested_file as path
                        tested_file_path = os.path.join(adjusted_repo_path, tested_file)
                    else:
                        # If path is specified, use it
                        tested_file_path = os.path.join(adjusted_repo_path, path)
                    
                    # Get the directory of the tested file
                    tested_file_dir = os.path.dirname(tested_file_path)
                    
                    # Ensure content is a list
                    if not isinstance(content_list, list):
                        content_list = [content_list]
                    
                    # Create a test file for each content item
                    for idx, content in enumerate(tqdm(content_list, desc=f"    Writing tests for {tested_file}", unit="test", leave=False)):
                        # Generate test filename
                        test_filename = generate_test_filename(tested_file, idx)
                        full_path = os.path.join(tested_file_dir, test_filename)
                        
                        # Calculate relative path for the test file from the repository root
                        if working_dir:
                            relative_path = str(Path(full_path).relative_to(adjusted_repo_path))
                        else:
                            relative_path = test_filename
                        
                        # Check if parent directory exists, create if needed
                        if not os.path.isdir(tested_file_dir):
                            if not dry_run:
                                try:
                                    os.makedirs(tested_file_dir, exist_ok=True)
                                    tqdm.write(f"  Created directory: {tested_file_dir}")
                                except OSError as e:
                                    tqdm.write(f"  Error creating directory {tested_file_dir}: {str(e)}")
                                    skipped_files += 1
                                    continue
                        
                        # Write the test file
                        if dry_run:
                            tqdm.write(f"  [DRY RUN] Would write test file: {full_path}")
                            tqdm.write(f"  Testing file: {tested_file}")
                            tqdm.write(f"  Dependencies: {', '.join(dependencies)}")
                            written_files += 1
                            
                            # Add test metadata
                            if convert_output:
                                test_metadata[repo_path]["tests"].append({
                                    "path": relative_path,
                                    "tested_files": tested_file  # Changed from list to string
                                })
                        else:
                            try:
                                with open(full_path, 'w') as test_f:
                                    test_f.write(content)
                                tqdm.write(f"  Wrote test file: {full_path}")
                                tqdm.write(f"  Testing file: {tested_file}")
                                tqdm.write(f"  Dependencies: {', '.join(dependencies)}")
                                written_files += 1
                                
                                # Add test metadata
                                if convert_output:
                                    test_metadata[repo_path]["tests"].append({
                                        "path": relative_path,
                                        "tested_files": tested_file  # Changed from list to string
                                    })
                            except Exception as e:
                                tqdm.write(f"  Error writing test file {full_path}: {str(e)}")
                                skipped_files += 1
            
            except json.JSONDecodeError:
                tqdm.write(f"Error: Line {line_num} is not valid JSON, skipping")
                continue
            except Exception as e:
                tqdm.write(f"Error processing line {line_num}: {str(e)}")
                continue
    
    # Print summary
    print("\n===== Summary =====")
    print(f"Processed repositories: {processed_repos}")
    print(f"Skipped repositories: {skipped_repos}")
    print(f"Test files written: {written_files}")
    print(f"Test files skipped: {skipped_files}")
    
    # Write converted format if requested
    if convert_output:
        write_converted_format(test_metadata, convert_output)
    
    return processed_repos, skipped_repos, written_files, skipped_files


def write_converted_format(test_metadata, output_file):
    """
    Write the test metadata in the format compatible with run_tests.py.
    
    Args:
        test_metadata: Dictionary mapping repository path to test metadata
        output_file: Path to the output JSONL file to be written
    """
    print(f"\nWriting test metadata to {output_file}")
    
    converted_count = 0
    with open(output_file, 'w') as out_f:
        for repo_path, metadata in test_metadata.items():
            if metadata["tests"]:  # Only write if there are tests
                out_f.write(json.dumps(metadata) + "\n")
                converted_count += 1
    
    print(f"Successfully wrote test metadata for {converted_count} repositories to {output_file}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Process the JSONL file, generating test files and conversion format if requested
    process_jsonl(args.input_file, args.working_dir, args.skip_pattern, args.dry_run, args.convert_output)


if __name__ == "__main__":
    main() 