#!/usr/bin/env python3

import argparse
import json
import os
import sys
import re
import multiprocessing
import concurrent.futures
from pathlib import Path
from functools import partial

# Global variable to track quiet mode
QUIET_MODE = False

def log(message, force=False):
    """Print message only if not in quiet mode or if force is True."""
    if not QUIET_MODE or force:
        print(message)

try:
    from tqdm import tqdm
except ImportError:
    log("Warning: tqdm module not found. Install it using 'pip install tqdm' for progress bars.")
    # Create a simple tqdm replacement that does nothing
    class FakeTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        
        def __iter__(self):
            return iter(self.iterable)
        
        def write(self, s, **kwargs):
            log(s)
    
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
    parser.add_argument("--workers", "-p", type=int, default=multiprocessing.cpu_count(),
                        help="Number of worker processes (default: number of CPU cores)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode - show only progress bar")
    return parser.parse_args()


def generate_test_filename(original_file, index):
    """Generate a test filename based on the original file name and index."""
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    return f"test_{base_name}_{index}.py"


def process_repo(repo_data_line, line_num, working_dir=None, skip_regex=None, dry_run=False, convert_output=False):
    """Process a single repository data line."""
    repo_results = {
        "processed": 0,
        "skipped": 0,
        "written_files": 0,
        "skipped_files": 0,
        "test_metadata": {}
    }
    
    try:
        # Parse the JSON object
        repo_data = json.loads(repo_data_line)
        repo_path = repo_data.get("repository")
        
        if not repo_path:
            log(f"Warning: Line {line_num} - Missing repository path, skipping")
            return repo_results
        
        # Check if repository matches skip pattern
        if skip_regex and skip_regex.search(repo_path):
            log(f"Skipping repository (matches skip pattern): {repo_path}")
            repo_results["skipped"] = 1
            return repo_results
        
        # Initialize test metadata for this repository
        if convert_output:
            repo_results["test_metadata"][repo_path] = {
                "repository": repo_path,
                "tests": []
            }
        
        # Adjust repository path if working_dir is provided
        adjusted_repo_path = repo_path
        if working_dir:
            adjusted_repo_path = os.path.join(working_dir, repo_path)
        
        # Check if repository directory exists
        if not os.path.isdir(adjusted_repo_path):
            log(f"Warning: Repository directory not found: {adjusted_repo_path}")
            # Count total test contents across all test files
            test_count = sum(len(test_info.get("content", [])) 
                            for test_info in repo_data.get("tests", {}).values() 
                            if isinstance(test_info.get("content", []), list))
            log(f"  Skipped {test_count} test files")
            repo_results["skipped_files"] = test_count
            return repo_results
        
        log(f"\nProcessing repository: {adjusted_repo_path}")
        repo_results["processed"] = 1
        
        # Process each tested file from the "tests" dictionary
        test_files = repo_data.get("tests", {})
        for tested_file, test_info in test_files.items():
            path = test_info.get("path")
            content_list = test_info.get("content", [])
            dependencies = test_info.get("dependencies", [])
            
            # Skip if content is empty
            if not content_list:
                log(f"  Warning: No test content for {tested_file}, skipping")
                repo_results["skipped_files"] += 1
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
            for idx, content in enumerate(content_list):
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
                            log(f"  Created directory: {tested_file_dir}")
                        except OSError as e:
                            log(f"  Error creating directory {tested_file_dir}: {str(e)}")
                            repo_results["skipped_files"] += 1
                            continue
                
                # Write the test file
                if dry_run:
                    log(f"  [DRY RUN] Would write test file: {full_path}")
                    log(f"  Testing file: {tested_file}")
                    log(f"  Dependencies: {', '.join(dependencies)}")
                    repo_results["written_files"] += 1
                    
                    # Add test metadata
                    if convert_output:
                        repo_results["test_metadata"][repo_path]["tests"].append({
                            "path": relative_path,
                            "tested_files": tested_file  # Changed from list to string
                        })
                else:
                    try:
                        with open(full_path, 'w') as test_f:
                            test_f.write(content)
                        log(f"  Wrote test file: {full_path}")
                        log(f"  Testing file: {tested_file}")
                        log(f"  Dependencies: {', '.join(dependencies)}")
                        repo_results["written_files"] += 1
                        
                        # Add test metadata
                        if convert_output:
                            repo_results["test_metadata"][repo_path]["tests"].append({
                                "path": relative_path,
                                "tested_files": tested_file  # Changed from list to string
                            })
                    except Exception as e:
                        log(f"  Error writing test file {full_path}: {str(e)}")
                        repo_results["skipped_files"] += 1
    
    except json.JSONDecodeError:
        log(f"Error: Line {line_num} is not valid JSON, skipping")
    except Exception as e:
        log(f"Error processing line {line_num}: {str(e)}")
    
    return repo_results


def process_jsonl(input_file, working_dir=None, skip_pattern=None, dry_run=False, convert_output=None, num_workers=None):
    """Process the input JSONL file and write test files using parallel processing."""
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
            log(f"Will skip repositories matching pattern: {skip_pattern}")
        except re.error as e:
            log(f"Warning: Invalid regex pattern '{skip_pattern}', ignoring. Error: {str(e)}")
    
    log(f"Processing repositories from {input_file}")
    
    # Count total lines first for the progress bar
    total_lines = sum(1 for _ in open(input_file, 'r'))
    
    # If no worker count is specified, use the number of CPU cores
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    log(f"Using {num_workers} worker processes")
    
    # Read all lines from the file
    with open(input_file, 'r') as f:
        lines_with_index = [(idx+1, line) for idx, line in enumerate(f)]
    
    # Create a partial function with common arguments
    process_repo_partial = partial(
        process_repo_with_args,
        working_dir=working_dir,
        skip_regex=skip_regex,
        dry_run=dry_run,
        convert_output=bool(convert_output)
    )
    
    # Process repositories in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the process_repo function to each line
        future_to_line = {
            executor.submit(process_repo_partial, line_idx, line): line_idx 
            for line_idx, line in lines_with_index
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_line), total=len(lines_with_index), desc="Processing repositories", unit="repo"):
            line_idx = future_to_line[future]
            try:
                result = future.result()
                
                # Update counters
                processed_repos += result["processed"]
                skipped_repos += result["skipped"]
                written_files += result["written_files"]
                skipped_files += result["skipped_files"]
                
                # Merge test metadata
                if convert_output and result["test_metadata"]:
                    test_metadata.update(result["test_metadata"])
            except Exception as e:
                log(f"Error processing line {line_idx}: {str(e)}")
    
    # Print summary
    log("\n===== Summary =====")
    log(f"Processed repositories: {processed_repos}")
    log(f"Skipped repositories: {skipped_repos}")
    log(f"Test files written: {written_files}")
    log(f"Test files skipped: {skipped_files}")
    
    # Write converted format if requested
    if convert_output:
        write_converted_format(test_metadata, convert_output)
    
    return processed_repos, skipped_repos, written_files, skipped_files


def process_repo_with_args(line_idx, line, working_dir, skip_regex, dry_run, convert_output):
    """Wrapper function to unpack arguments for process_repo."""
    return process_repo(line, line_idx, working_dir, skip_regex, dry_run, convert_output)


def write_converted_format(test_metadata, output_file):
    """
    Write the test metadata in the format compatible with run_tests.py.
    
    Args:
        test_metadata: Dictionary mapping repository path to test metadata
        output_file: Path to the output JSONL file to be written
    """
    log(f"\nWriting test metadata to {output_file}")
    
    converted_count = 0
    with open(output_file, 'w') as out_f:
        for repo_path, metadata in test_metadata.items():
            if metadata["tests"]:  # Only write if there are tests
                out_f.write(json.dumps(metadata) + "\n")
                converted_count += 1
    
    log(f"Successfully wrote test metadata for {converted_count} repositories to {output_file}")


def main():
    """Main entry point."""
    global QUIET_MODE
    args = parse_arguments()
    
    # Set quiet mode based on args
    QUIET_MODE = args.quiet
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        log(f"Error: Input file '{args.input_file}' does not exist", force=True)
        sys.exit(1)
    
    # Process the JSONL file, generating test files and conversion format if requested
    process_jsonl(
        args.input_file, 
        args.working_dir, 
        args.skip_pattern, 
        args.dry_run, 
        args.convert_output,
        args.workers
    )


if __name__ == "__main__":
    main() 