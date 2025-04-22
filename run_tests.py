#!/usr/bin/env python3
"""
Simplified version of Repo2Run that only implements the --run-tests functionality.

This script reads test input files (default: test.jsonl) containing previously extracted test files
and runs the tests using the current Python environment. It handles both verbose
and non-verbose modes.

Usage:
    python run_tests.py [--input-file input.jsonl] [--output-file results.jsonl] [--verbose] [--num-workers N]

Options:
    --input-file FILE      Input file name (default: test.jsonl in current directory)
    --output-file FILE     Output file name (default: test_results.jsonl in current directory)
    --verbose              Enable verbose logging
    --num-workers N        Number of worker processes for parallel processing (default: number of CPU cores)
    --timeout SECONDS      Timeout in seconds (default: 1800 - 0.5 hour)
    --require-tested-files Only run test files that have tested_files information
    --skip-pytest-flags    Skip restrictive pytest flags (--noconftest, -o addopts="") for tests related to pytest
    --use-separate-envs    Create separate virtual environments for each repository
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import datetime
import multiprocessing
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
import tempfile
import concurrent.futures
from tqdm import tqdm
import signal
import psutil
import threading
import functools
import traceback
import re
import venv

# Modify import to work from any directory
# First determine the script's directory to use as base for importing
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Add the script's directory to Python path
sys.path.insert(0, str(SCRIPT_DIR))

try:
    # Try the relative import first (assuming we're in the test-runner directory)
    from utils.logger import configure_process_logging
except ImportError:
    # If that fails, define the function directly here as a fallback
    def configure_process_logging(verbose: bool) -> logging.Logger:
        """Configure logging for a worker process.
        
        Args:
            verbose (bool): Whether to enable verbose logging. When disabled, no logs will be shown.
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('test_runner')
        logger.setLevel(logging.INFO if verbose else logging.CRITICAL)  # Only show logs in verbose mode
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add a new handler that only shows logs if verbose is True
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO if verbose else logging.CRITICAL)  # Only show logs in verbose mode
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Make sure we don't propagate to the root logger, which might have different settings
        logger.propagate = False
        
        return logger


def create_virtual_env(repo_path, logger):
    """Create a minimal virtual environment for a repository without installing dependencies.
    
    Args:
        repo_path (Path): Path to the repository
        logger (logging.Logger): Logger instance
    
    Returns:
        Path: Path to the virtual environment Python executable or None if creation fails
    """
    try:
        # Create a temporary directory for the virtual environment
        venv_dir = tempfile.mkdtemp(prefix=f"venv_{repo_path.name}_")
        venv_path = Path(venv_dir)
        
        logger.info(f"Creating minimal virtual environment at {venv_path}")
        
        # Create virtual environment
        venv.create(venv_path, with_pip=True)
        
        # Determine the Python executable path in the virtual environment
        if os.name == 'nt':  # Windows
            venv_python = venv_path / 'Scripts' / 'python.exe'
            pip_exec = venv_path / 'Scripts' / 'pip.exe'
        else:  # Unix/Linux/MacOS
            venv_python = venv_path / 'bin' / 'python'
            pip_exec = venv_path / 'bin' / 'pip'
        
        if not venv_python.exists():
            logger.error(f"Failed to find Python executable in virtual environment: {venv_python}")
            return None
        
        logger.info(f"Created virtual environment with Python at {venv_python}")
        
        # Install hydra-core in the virtual environment
        logger.info(f"Installing hydra-core in virtual environment at {venv_path}")
        try:
            subprocess.check_call([str(pip_exec), "install", "hydra-core"], 
                                 stdout=subprocess.PIPE if not logger.level <= logging.INFO else None,
                                 stderr=subprocess.PIPE if not logger.level <= logging.INFO else None)
            logger.info("Successfully installed hydra-core")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install hydra-core: {e}")
            # Continue even if installation fails - the environment is still usable
        
        # Return the virtual environment Python executable path
        return venv_python
    
    except Exception as e:
        logger.error(f"Error creating virtual environment: {e}")
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run tests from previously extracted test input files.'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default='test.jsonl',
        help='Input file name (default: test.jsonl)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='test_results.jsonl',
        help='Output file name (default: test_results.jsonl)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=os.cpu_count(),
        help='Number of worker processes for parallel processing (default: number of CPU cores)'
    )
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=1800,
        help='Timeout in seconds (default: 1800 - 0.5 hour)'
    )
    parser.add_argument(
        '--require-tested-files',
        action='store_true',
        help='Only run test files that have tested_files information'
    )
    parser.add_argument(
        '--skip-pytest-flags',
        action='store_true',
        help='Skip restrictive pytest flags (--noconftest, -o addopts="") for tests related to pytest itself'
    )
    parser.add_argument(
        '--use-separate-envs',
        action='store_true',
        help='Create separate virtual environments for each repository'
    )
    
    return parser.parse_args()


def run_tests_from_jsonl(args: argparse.Namespace) -> int:
    """
    Run tests from previously extracted test input file without creating virtual environments.
    Tests will be run using pytest with the system Python interpreter directly in the repository directories
    where the tests were originally located.
    
    Args:
        args: Command line arguments
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Configure logging
    logger = configure_process_logging(args.verbose)
    
    # Handle file paths - working from current directory
    test_jsonl_path = Path(args.input_file)
    test_results_jsonl_path = Path(args.output_file)
    
    # Check if input file exists
    if not test_jsonl_path.exists():
        logger.error(f"Test file {test_jsonl_path} does not exist. Run extraction first.")
        return 1
    
    # Create or clear test_results.jsonl file
    if test_results_jsonl_path.exists():
        logger.info(f"Clearing existing test results file: {test_results_jsonl_path}")
        with open(test_results_jsonl_path, 'w') as f:
            pass
    
    # Track repositories with their test data
    repositories_data = []
    
    # Read repositories from input file
    logger.info(f"Reading test data from {test_jsonl_path}")
    with open(test_jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    repositories_data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {args.input_file}: {e}")
    
    logger.info(f"Found {len(repositories_data)} repositories with test data")
    
    # Check if there are no repositories found
    if len(repositories_data) == 0:
        logger.info(f"No repositories found in {args.input_file}")
        return 1
    
    # Set require_tested_files flag
    if args.require_tested_files:
        logger.info("Require tested_files mode is enabled - will skip test files without tested_files information")
    
        # Filter repositories to only include those with valid test files
        filtered_repositories_data = []
        for repo_data in repositories_data:
            repo_identifier = repo_data.get("repository", "")
            tests = repo_data.get("tests", [])
            
            # Filter tests to only include those with non-empty tested_files
            valid_tests = [test for test in tests if test.get("tested_files")]
            
            if valid_tests:
                # Update the repo data with only valid tests
                filtered_repo_data = repo_data.copy()
                filtered_repo_data["tests"] = valid_tests
                filtered_repositories_data.append(filtered_repo_data)
                logger.info(f"Repository {repo_identifier}: Kept {len(valid_tests)} out of {len(tests)} test files with tested_files information")
            else:
                logger.info(f"Repository {repo_identifier}: Skipped - no test files with tested_files information")
        
        # Update repositories_data with filtered data
        skipped_count = len(repositories_data) - len(filtered_repositories_data)
        logger.info(f"Filtered out {skipped_count} repositories with no valid test files")
        repositories_data = filtered_repositories_data
        
        if not repositories_data:
            logger.info("No repositories with valid test files (containing tested_files information) found")
            return 0
    
    # Convert repository data to paths for run_tests_parallel
    repositories = []
    repo_test_info = {}  # Store test info by repository path for later use
    
    for repo_data in repositories_data:
        repo_identifier = repo_data.get("repository", "")
        if repo_identifier:
            # For local paths, we need to check if they exist
            repo_path = Path(repo_identifier)
            if repo_path.exists():
                tests_info = repo_data.get("tests", [])
                if tests_info:
                    # Validate test file paths
                    valid_test_files = []
                    for test_info in tests_info:
                        if "path" in test_info:
                            test_path = test_info.get("path")
                            # Make a copy of the test info to modify
                            processed_test_info = test_info.copy()
                            
                            # Get tested_files field
                            tested_files = processed_test_info.get("tested_files", [])
                            
                            # Skip test files without tested_files if require_tested_files is enabled
                            if args.require_tested_files and not tested_files:
                                logger.debug(f"Skipping test file without tested_files: {test_path}")
                                continue
                            
                            # Check if the test file exists
                            full_path = repo_path / test_path
                            if full_path.exists():
                                valid_test_files.append(processed_test_info)
                                logger.debug(f"Test file found: {test_path} with {len(tested_files)} tested files")
                            else:
                                logger.warning(f"Test file not found: {test_path} in repository {repo_identifier}")
                        else:
                            logger.warning(f"Test info missing path in repository {repo_identifier}")
                    
                    if valid_test_files:
                        # Only add repository if it has valid test files after filtering
                        repositories.append(repo_path)
                        repo_test_info[str(repo_path)] = valid_test_files
                        logger.info(f"Found {len(valid_test_files)} valid test files for repository {repo_identifier}")
                    else:
                        logger.warning(f"No valid test files found for repository {repo_identifier}")
                else:
                    logger.warning(f"No test files defined for repository {repo_identifier}")
            else:
                logger.warning(f"Repository directory {repo_path} does not exist. Skipping.")
    
    if not repositories:
        logger.error(f"No valid repository paths found in {test_jsonl_path.name}. Exiting.")
        return 1
    
    logger.info(f"Running tests for {len(repositories)} repositories in parallel")
    
    # Determine whether to use separate virtual environments
    unified_venv = None
    if args.use_separate_envs:
        logger.info("Using separate virtual environments for each repository")
    else:
        logger.info("Using current Python environment for all repositories")
    
    # Run tests in parallel
    test_results = run_tests_parallel(repositories, None, unified_venv, args, repo_test_info, logger)

    # Write test results to jsonl file
    results_written = 0
    logger.info(f"Writing test results to {test_results_jsonl_path}")
    with open(test_results_jsonl_path, 'w') as f:
        for result in test_results:
            f.write(json.dumps(result) + "\n")
            results_written += 1
    
    logger.info(f"Finished writing {results_written} test results")
    return 0


def run_tests_parallel(repositories, output_dir, unified_venv, args, repo_test_info, logger):
    """
    Run tests for all repositories in parallel.
    
    Args:
        repositories: List of repositories
        output_dir: Output directory
        unified_venv: Path to the unified virtual environment (None to use current environment)
        args: Command line arguments
        repo_test_info: Dictionary mapping repository paths to test file information
        logger: Logger instance
    
    Returns:
        list: Test results for all repositories
    """
    logger.info(f"🧪 Running tests for {len(repositories)} repositories")
    
    # Print environment info
    logger.info(f"📂 Environment details:")
    logger.info(f"  - Using current Python environment: {sys.executable}")
    logger.info(f"  - Python version: {sys.version.split()[0]}")
    
    # Use the number of workers specified
    max_workers = args.num_workers
    available_cores = multiprocessing.cpu_count()
    suggested_workers = min(available_cores, len(repositories), max_workers)
    max_workers = suggested_workers
    logger.info(f"Using {max_workers} worker processes for parallel test execution")
    
    def kill_process_tree(pid):
        """Kill a process and all its children."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Send SIGTERM to children first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Send SIGTERM to parent
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass
            
            # Wait for processes to terminate
            gone, alive = psutil.wait_procs(children + [parent], timeout=3)
            
            # If any processes are still alive, send SIGKILL
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
                
            # Double check if parent is still alive
            try:
                if parent.is_running():
                    parent.kill()  # Force kill if still running
            except psutil.NoSuchProcess:
                pass
            
            # Double check children
            for child in children:
                try:
                    if child.is_running():
                        child.kill()  # Force kill if still running
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            logger.warning(f"Error killing process tree {pid}: {e}")
    
    def cleanup_process(future, repo_name=None):
        """Clean up a process associated with a future."""
        if not future.done():
            try:
                # Get the process ID if available
                if hasattr(future, '_process'):
                    pid = future._process.pid
                    if repo_name:
                        logger.warning(f"Killing process tree for repository {repo_name} (PID: {pid})")
                    kill_process_tree(pid)
                
                # Cancel the future
                future.cancel()
                
                # Wait a short time for cancellation to take effect
                time.sleep(0.1)
                
                # Force cancel again if still not done
                if not future.done():
                    future.cancel()
            except Exception as e:
                logger.warning(f"Error cleaning up process: {e}")
    
    # Create a threading Event for signaling shutdown
    shutdown_event = threading.Event()
    
    def worker_init():
        """Initialize worker process"""
        # Ignore SIGINT in worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Set up process group for easier cleanup
        os.setpgrp()
    
    # Process repositories in parallel
    all_results = []
    executor = None
    
    try:
        # Create a partial function with fixed arguments
        process_repo = functools.partial(
            process_repository,
            output_dir=output_dir,
            unified_venv=unified_venv,
            args=args,
            repo_test_info=repo_test_info
        )
        
        # Use a process pool for test running to enable proper timeout handling
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, 
            initializer=worker_init
        )
        
        # Submit all tasks
        future_to_repo = {}
        for repo in repositories:
            if shutdown_event.is_set():
                break
            future = executor.submit(process_repo, repo)
            future_to_repo[future] = repo
        
        # Create a progress bar with more information
        with tqdm(total=len(repositories), desc="Running tests", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            completed_futures = set()
            while len(completed_futures) < len(future_to_repo):
                if shutdown_event.is_set():
                    break
                
                # Check for completed futures
                newly_completed = {f for f in future_to_repo if f.done() and f not in completed_futures}
                
                for future in newly_completed:
                    repo = future_to_repo[future]
                    completed_futures.add(future)
                    
                    try:
                        # Display which repository is currently being processed
                        repo_name = repo.name if isinstance(repo, Path) else repo
                        pbar.set_postfix_str(f"Processing {repo_name}")
                        
                        result = future.result()
                        all_results.append(result)
                        
                        # Update progress
                        pbar.update(1)
                    except Exception as e:
                        # Handle error
                        logger.error(f"Error processing repository {repo}: {e}")
                        repo_str = str(repo)
                        all_results.append({
                            "repository": repo_str,
                            "status": "error",
                            "error": str(e),
                            "execution": {
                                "start_time": time.time(),
                                "elapsed_time": 0
                            },
                            "tests": {
                                "found": 0,
                                "passed": 0,
                                "failed": 0,
                                "skipped": 0, 
                                "details": []
                            }
                        })
                        
                        # Update progress
                        pbar.update(1)
                
                # If no newly completed futures, sleep a bit
                if not newly_completed:
                    time.sleep(0.1)
        
        # Check for any futures that have not completed yet
        for future, repo in future_to_repo.items():
            if future not in completed_futures:
                repo_name = repo.name if isinstance(repo, Path) else repo
                logger.warning(f"Repository {repo_name} timed out or was not processed")
                cleanup_process(future, repo_name)
                
                # Add timeout result
                all_results.append({
                    "repository": str(repo),
                    "status": "error",
                    "error": "Timeout or execution was interrupted",
                    "execution": {
                        "start_time": time.time(),
                        "elapsed_time": args.timeout
                    },
                    "tests": {
                        "found": 0,
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "details": []
                    }
                })
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received, shutting down...")
        shutdown_event.set()
        
        # Cancel all futures
        for future, repo in future_to_repo.items():
            if not future.done():
                repo_name = repo.name if isinstance(repo, Path) else repo
                cleanup_process(future, repo_name)
    finally:
        # Shutdown the executor properly
        if executor:
            executor.shutdown(wait=False)
    
    return all_results


def process_repository(repo, output_dir, unified_venv, args, repo_test_info=None):
    """Process a single repository and handle errors.
    
    Args:
        repo: Repository path
        output_dir: Output directory path
        unified_venv: Virtual environment path (None to use current environment)
        args: Command line arguments
        repo_test_info: Dictionary mapping repository paths to test file information
    
    Returns:
        dict: Test results for the repository
    """
    try:
        # Configure logger with the correct verbose setting
        logger = configure_process_logging(args.verbose)
        
        test_file_list = None
        test_file_metadata = None
        
        if repo_test_info:
            repo_key = str(repo)
            if repo_key in repo_test_info:
                test_files_info = repo_test_info[repo_key]
                if test_files_info:
                    test_file_list = []
                    test_file_metadata = {}
                    for test_info in test_files_info:
                        if "path" in test_info:
                            test_path = test_info["path"]
                            test_file_list.append(test_path)
                            test_file_metadata[test_path] = {
                                "tested_files": test_info.get("tested_files", [])
                            }
        
        # Run tests for the repository
        return run_tests_for_repo(repo, output_dir, unified_venv, args, 
                               test_file_list=test_file_list,
                               test_file_metadata=test_file_metadata)
    except Exception as e:
        # Create an error result
        repo_str = str(repo)
        return {
            "repository": repo_str,
            "status": "error",
            "error": str(e),
            "tests": {"found": 0, "passed": 0, "failed": 0, "skipped": 0, "details": []},
            "execution": {
                "start_time": time.time(),
                "elapsed_time": 0
            }
        }


def run_tests_for_repo(repo_path, output_dir, unified_venv, args, test_file_list=None, test_file_metadata=None):
    """
    Run tests for a single repository.
    
    Args:
        repo_path: Path to the repository
        output_dir: Output directory
        unified_venv: Path to the virtual environment (None to use current environment)
        args: Command line arguments
        test_file_list: Optional list of test file paths
        test_file_metadata: Optional dictionary with metadata for test files
    
    Returns:
        dict: Test results
    """
    # Configure logger with the correct verbose setting
    logger = configure_process_logging(args.verbose)
    
    result_data = {
        "repository": str(repo_path),
        "status": "running",
        "execution": {
            "start_time": time.time(),
            "elapsed_time": 0
        },
        "tests": {
            "found": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
    }
    
    def add_log_entry(message, level="INFO", **kwargs):
        """Add a log entry to the logger only."""
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
    
    venv_python = None
    env_temp_dir = None
    
    try:
        # Set working directory to repository path
        working_dir = repo_path
        add_log_entry(f"Working directory set to {working_dir}")
        
        # Create a virtual environment if requested
        if args.use_separate_envs:
            add_log_entry(f"Creating virtual environment for repository {repo_path.name}")
            venv_python = create_virtual_env(repo_path, logger)
            if venv_python:
                add_log_entry(f"Using virtual environment Python: {venv_python}")
            else:
                add_log_entry("Failed to create virtual environment, falling back to system Python", level="WARNING")
                venv_python = sys.executable
        else:
            # Use the system Python interpreter
            venv_python = sys.executable
        
        # Process test files if provided
        test_files = None
        test_metadata = {}  # Will store metadata for each test file
        
        if test_file_list:
            # Convert test file paths to Path objects relative to working_dir
            test_files = []
            skipped_files = []
            
            # Check if we should require tested_files
            require_tested_files = args.require_tested_files
            
            for test_file_path in test_file_list:
                try:
                    # Check if this test file has tested_files metadata
                    if require_tested_files and test_file_metadata and test_file_path in test_file_metadata:
                        metadata = test_file_metadata[test_file_path]
                        tested_files = metadata.get("tested_files", [])
                        
                        # Skip test files without tested_files information
                        if not tested_files:
                            add_log_entry(f"Skipping test file {test_file_path} because it doesn't have tested_files information", level="WARNING")
                            skipped_files.append(test_file_path)
                            continue
                    
                    # Try multiple ways to locate the test file
                    # 1. Direct path from working directory
                    file_path = working_dir / test_file_path
                    
                    # 2. If not found, try with just the filename
                    if not file_path.exists():
                        file_path = working_dir / Path(test_file_path).name
                    
                    # 3. Try a glob pattern based on the filename
                    if not file_path.exists():
                        file_name = Path(test_file_path).name
                        glob_pattern = f"**/{file_name}"
                        glob_results = list(working_dir.glob(glob_pattern))
                        if glob_results:
                            file_path = glob_results[0]  # Use the first match
                    
                    # Add the file if it exists
                    if file_path.exists():
                        test_files.append(file_path)
                        relative_path = file_path.relative_to(working_dir)
                        add_log_entry(f"Found test file: {relative_path}")
                        
                        # Capture metadata for this test file if available
                        if test_file_metadata and test_file_path in test_file_metadata:
                            # Store with both absolute and relative paths for easier lookup
                            test_metadata[str(file_path)] = test_file_metadata[test_file_path]
                            test_metadata[str(relative_path)] = test_file_metadata[test_file_path]
                    else:
                        add_log_entry(f"Test file not found: {test_file_path}", level="WARNING")
                except Exception as e:
                    add_log_entry(f"Error processing test file path {test_file_path}: {str(e)}", level="WARNING")
            
            if test_files:
                add_log_entry(f"Using {len(test_files)} test files from input file")
            else:
                add_log_entry("None of the provided test files were found", level="WARNING")
                result_data["status"] = "skipped"
                result_data["error"] = "No test files found"
                result_data["execution"]["elapsed_time"] = time.time() - result_data["execution"]["start_time"]
                return result_data
            
            # Add information about skipped files to result_data
            if skipped_files:
                add_log_entry(f"Skipped {len(skipped_files)} test files due to missing tested_files information", level="WARNING")
                result_data["skipped_files"] = {
                    "count": len(skipped_files),
                    "files": skipped_files,
                    "reason": "Missing tested_files information"
                }
        else:
            # No test files provided
            add_log_entry("No test files provided", level="WARNING")
            result_data["status"] = "skipped"
            result_data["error"] = "No test files provided"
            result_data["execution"]["elapsed_time"] = time.time() - result_data["execution"]["start_time"]
            return result_data
        
        # Run pytest on the test files
        if test_files:
            # print the working directory
            add_log_entry(f"Working directory: {working_dir}")
            add_log_entry(f"Running pytest on {len(test_files)} test files")
            
            # Prepare test files paths for pytest - use paths relative to working directory
            test_file_paths = [str(f.relative_to(working_dir)) for f in test_files]
            add_log_entry(f"Test file paths: {test_file_paths}")
            # Run pytest with XML output for parsing results
            xml_output_file = tempfile.mktemp(suffix=".xml")
            
            # Construct pytest command using the virtual environment Python if available
            python_exec = venv_python
            
            add_log_entry(f"Using Python executable: {python_exec}")
            
            # Check if any test files are related to pytest itself
            pytest_related_tests = False
            for test_path in test_file_paths:
                # Check if the test is related to pytest based on filename or metadata
                if "pytest" in test_path.lower() or (test_metadata and any("pytest" in str(meta).lower() for meta in test_metadata.values())):
                    pytest_related_tests = True
                    add_log_entry(f"Detected pytest-related test: {test_path}", level="INFO")
                    break
            
            # Base command that's always used
            cmd = [
                str(python_exec), "-m", "pytest",
                *test_file_paths,
                "-v",
                f"--junitxml={xml_output_file}",
                "--color=no"
            ]
            
            # Add restrictive flags only if not testing pytest itself
            if not (pytest_related_tests or args.skip_pytest_flags):
                cmd.extend([
                    "--noconftest",
                    "-o",
                    "addopts=''",
                    "--continue-on-collection-errors",
                ])
            else:
                add_log_entry("Running with modified pytest flags to allow pytest testing", level="INFO")
            
            add_log_entry(f"Running command: {' '.join(cmd)}")
            
            # Prepare environment variables
            env_vars = os.environ.copy()
            
            # If using a separate virtual environment, set PYTHONPATH to include system paths
            if args.use_separate_envs and venv_python and venv_python != sys.executable:
                # Create a PYTHONPATH that includes the current sys.path
                python_path = os.pathsep.join(sys.path)
                # Add the repository path to PYTHONPATH
                if python_path:
                    python_path = f"{str(repo_path)}{os.pathsep}{python_path}"
                else:
                    python_path = str(repo_path)
                
                env_vars["PYTHONPATH"] = python_path
                add_log_entry(f"Setting PYTHONPATH to: {python_path}", level="INFO")
            
            try:
                # Set timeout for subprocess
                timeout = args.timeout
                
                # Run pytest
                start_time = time.time()
                
                process = subprocess.run(
                    cmd,
                    cwd=str(working_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env_vars
                )
                
                elapsed_time = time.time() - start_time
                
                # Parse the output
                stdout = process.stdout
                stderr = process.stderr
                
                # Check if XML file was generated
                xml_results = []
                test_case_status_map = {}  # New dictionary to map test case names to status
                if os.path.exists(xml_output_file):
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(xml_output_file)
                        root = tree.getroot()
                        
                        # Log metadata information for debugging
                        if test_metadata:
                            add_log_entry(f"Found test metadata for {len(test_metadata)} test files", level="INFO")
                            for test_path, metadata in test_metadata.items():
                                if "group" in metadata:
                                    group_count = len(metadata["group"])
                                    add_log_entry(f"Test file {test_path} has {group_count} groups", level="INFO")
                        
                        # Extract test cases
                        for testcase in root.findall(".//testcase"):
                            test_name = testcase.get("name")
                            class_name = testcase.get("classname")
                            
                            # Log test case information
                            add_log_entry(f"Processing test case: {class_name}.{test_name}", level="INFO")
                            
                            # Create test data object
                            test_data = {
                                "name": test_name,
                                "classname": class_name,
                                "time": float(testcase.get("time", 0)),
                                "status": "passed"
                            }
                            
                            # Check for failures or errors
                            failures = testcase.findall("failure")
                            errors = testcase.findall("error")
                            skipped = testcase.findall("skipped")
                            
                            if failures:
                                test_data["status"] = "failed"
                                test_data["message"] = failures[0].get("message", "")
                                test_data["traceback"] = failures[0].text
                            elif errors:
                                test_data["status"] = "error"
                                test_data["message"] = errors[0].get("message", "")
                                test_data["traceback"] = errors[0].text
                            elif skipped:
                                test_data["status"] = "skipped"
                                test_data["message"] = skipped[0].get("message", "")
                            
                            xml_results.append(test_data)
                            
                            # Group test cases by test file
                            # Extract the file name from the class_name (assumed format: test_manage_X.module.TestClass)
                            class_parts = class_name.split('.')
                            if len(class_parts) > 0:
                                # Find the test file name (e.g., test_manage_X)
                                file_part = ".".join(class_parts[:-1])
                                
                                if file_part:
                                    # Extract the actual class name (last part after the last dot)
                                    actual_class_name = class_parts[-1] if class_parts else class_name
                                    
                                    # Create a key using just the class name and test name
                                    test_case_key = f"{actual_class_name}.{test_name}"
                                    
                                    # Initialize the dictionary for this test file if it doesn't exist
                                    if file_part not in test_case_status_map:
                                        test_case_status_map[file_part] = {}
                                    
                                    # Add the test case status to the dictionary
                                    test_case_status_map[file_part][test_case_key] = test_data["status"]
                                    
                                    add_log_entry(f"Mapped test case: {class_name}.{test_name} -> {file_part}: {test_case_key} [{test_data['status']}]", level="INFO")
                    except Exception as e:
                        add_log_entry(f"Error parsing XML results: {str(e)}", level="ERROR")
                
                # Remove temp XML file
                try:
                    if os.path.exists(xml_output_file):
                        os.unlink(xml_output_file)
                except Exception:
                    pass
                
                # Parse test results from stdout
                if process.returncode == 0:
                    # All tests passed
                    result_data["status"] = "success"
                else:
                    # Some tests failed
                    result_data["status"] = "failure"
                
                # Update result data
                result_data["execution"]["elapsed_time"] = elapsed_time
                result_data["execution"]["stdout"] = stdout
                result_data["execution"]["stderr"] = stderr
                result_data["execution"]["exit_code"] = process.returncode
                
                # Add test details
                found_tests = len(xml_results)
                passed_tests = sum(1 for t in xml_results if t["status"] == "passed")
                failed_tests = sum(1 for t in xml_results if t["status"] in ["failed", "error"])
                skipped_tests = sum(1 for t in xml_results if t["status"] == "skipped")
                
                result_data["tests"]["found"] = found_tests
                result_data["tests"]["passed"] = passed_tests
                result_data["tests"]["failed"] = failed_tests
                result_data["tests"]["skipped"] = skipped_tests
                
                # Add test case status map to the result
                result_data["tests"]["test_case_status_map"] = test_case_status_map
                
                # Log summary of test case mapping
                add_log_entry(f"Test case mapping summary: mapped {sum(len(cases) for cases in test_case_status_map.values())} test cases", level="INFO")
                
                # Count total passed, failed, and skipped tests across all test files
                total_passed = 0
                total_failed = 0
                total_skipped = 0
                
                for test_file, cases in test_case_status_map.items():
                    file_passed = sum(1 for status in cases.values() if status == 'passed')
                    file_failed = sum(1 for status in cases.values() if status in ['failed', 'error'])
                    file_skipped = sum(1 for status in cases.values() if status == 'skipped')
                    
                    total_passed += file_passed
                    total_failed += file_failed
                    total_skipped += file_skipped
                    
                    add_log_entry(f"  - {test_file}: {len(cases)} tests ({file_passed} passed, {file_failed} failed, {file_skipped} skipped)", level="INFO")
                
                add_log_entry(f"  - Total: Passed: {total_passed}, Failed: {total_failed}, Skipped: {total_skipped}", level="INFO")
                
                # Add test details for pytest
                result_data["tests"]["details"] = [{
                    "name": "pytest",
                    "found": found_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "skipped": skipped_tests,
                    "tests": xml_results
                }]
                
            except subprocess.TimeoutExpired:
                add_log_entry(f"Timeout expired after {timeout} seconds", level="ERROR")
                result_data["status"] = "error"
                result_data["error"] = f"Timeout expired after {timeout} seconds"
                result_data["execution"]["elapsed_time"] = timeout
                
            except Exception as e:
                add_log_entry(f"Error running pytest: {str(e)}", level="ERROR")
                result_data["status"] = "error"
                result_data["error"] = f"Error running pytest: {str(e)}"
                result_data["execution"]["elapsed_time"] = time.time() - result_data["execution"]["start_time"]
        
        return result_data
    
    except Exception as e:
        # Handle any other exceptions
        add_log_entry(f"Unexpected error: {str(e)}", level="ERROR")
        result_data["status"] = "error"
        result_data["error"] = f"Unexpected error: {str(e)}"
        result_data["execution"]["elapsed_time"] = time.time() - result_data["execution"]["start_time"]
        return result_data
    
    finally:
        # Clean up virtual environment directory if we created one
        if args.use_separate_envs and venv_python and venv_python != sys.executable:
            try:
                # Get the virtual environment directory (parent of bin/python)
                venv_dir = Path(venv_python).parent.parent
                add_log_entry(f"Cleaning up virtual environment at {venv_dir}")
                shutil.rmtree(venv_dir, ignore_errors=True)
            except Exception as e:
                add_log_entry(f"Error cleaning up virtual environment: {e}", level="WARNING")


def main():
    """Main entry point for the program."""
    try:
        args = parse_arguments()
        
        # Run tests from input file
        return_code = run_tests_from_jsonl(args)
        
        # Force cleanup of any remaining processes
        try:
            # Get all child processes
            current_process = psutil.Process()
            
            # First try graceful termination
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Give processes time to terminate
            _, still_alive = psutil.wait_procs(children, timeout=3)
            
            # Force kill any remaining processes
            for child in still_alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            # Double check for any new children
            for child in current_process.children(recursive=True):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            # Force cleanup of multiprocessing resources
            try:
                multiprocessing.current_process()._cleanup()
            except:
                pass
                
            # Clean up multiprocessing queues
            try:
                for q in multiprocessing.active_children():
                    try:
                        q.terminate()
                    except:
                        pass
            except:
                pass
        except:
            pass
        
        # Give a moment for logging to complete
        time.sleep(0.1)
        return return_code
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 