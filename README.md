# Test Runner

A simplified version of Repo2Run that implements only the `--run-tests` functionality. This tool reads previously extracted test files from a `test.jsonl` file and runs them using the system Python interpreter.

## Features

- Run tests from previously extracted `test.jsonl` files
- Handle both verbose and non-verbose logging modes
- Run tests in parallel using multiple worker processes
- Timeout for long-running tests
- Generate detailed test results in `test_results.jsonl`
- Copy repositories from HDFS to local directory
- Group tests by their tested modules
- Map test results to grouped tests for better analysis

## Requirements

- Python 3.6+
- pytest
- tqdm (for progress bars)
- psutil (for process management)
- venv (for optional separate virtual environments)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/test-runner.git
   cd test-runner
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Script Overview

The test runner consists of three main scripts:

### run_tests.py

The main test runner script that executes tests from previously extracted JSONL files.

```bash
python run_tests.py [--input-file INPUT.jsonl] [--output-file RESULTS.jsonl] [--verbose] 
                   [--num-workers N] [--timeout SECONDS] [--require-tested-files] 
                   [--skip-pytest-flags] [--use-separate-envs]
```

#### Arguments

- `--input-file FILE`: Input file name (default: test.jsonl in current directory)
- `--output-file FILE`: Output file name (default: test_results.jsonl in current directory)
- `--verbose`: Enable verbose logging
- `--num-workers N`: Number of worker processes for parallel processing (default: number of CPU cores)
- `--timeout SECONDS`: Timeout in seconds (default: 1800 - 0.5 hour)
- `--require-tested-files`: Only run test files that have `tested_files` information
- `--skip-pytest-flags`: Skip restrictive pytest flags for tests related to pytest itself
- `--use-separate-envs`: Create separate virtual environments for each repository

### hdfs_repo_copy.py

A utility script for copying repositories from HDFS to a local directory.

```bash
python hdfs_repo_copy.py JSONL_FILES [--output-dir DIR] [--skip-existing] [--dry-run] 
                         [--verbose] [--max-repos N] [--workers N]
```

#### Arguments

- `JSONL_FILES`: Path(s) to the JSONL file(s) containing repository information
- `--output-dir DIR`: Local directory to copy repositories to (default: current directory)
- `--skip-existing`: Skip repositories that already exist locally
- `--dry-run`: Do not perform actual copying, just simulate the process
- `--verbose`: Print verbose output
- `--max-repos N`: Maximum number of repositories to process
- `--workers N`: Number of parallel workers (default: all available CPU cores)

### group_and_map_tests.py

A script for grouping tests and mapping test results to tested modules.

```bash
python group_and_map_tests.py input_file test_runner_output output_file
```

#### Arguments

- `input_file`: Path to the input JSONL file containing test information
- `test_runner_output`: Path to the test runner output JSONL file
- `output_file`: Path where to write the processed output file

## Input Format

The tool reads from a `test.jsonl` file in the specified directory. This file should contain one JSON object per line, with each object representing a repository and its test files.

Example `test.jsonl` format:
```json
{"repository": "/path/to/repo", "tests": [{"path": "tests/test_file.py", "tested_files": ["src/file.py"]}]}
```

## Output Format

Test results are written to a `test_results.jsonl` file, with one JSON object per line for each repository.

Example `test_results.jsonl` format:
```json
{
  "repository": "/path/to/repo",
  "status": "success",
  "execution": {
    "start_time": 1629123456.789,
    "elapsed_time": 2.34,
    "stdout": "...",
    "stderr": "...",
    "exit_code": 0
  },
  "tests": {
    "found": 5,
    "passed": 5,
    "failed": 0,
    "skipped": 0,
    "details": [{
      "name": "pytest",
      "found": 5,
      "passed": 5,
      "failed": 0,
      "skipped": 0,
      "tests": [...]
    }],
    "test_case_status_map": {
      "test_module": {
        "TestClass.test_method": "passed"
      }
    }
  }
}
```

## Grouped Test Output

After running `group_and_map_tests.py`, the output will include additional grouping information:

```json
{
  "repository": "/path/to/repo",
  "tests": [{
    "path": "tests/test_file.py",
    "group": {
      "ModuleName": ["0.TestClass.test_method1", "0.TestClass.test_method2"],
      "function_name": ["1.TestOtherClass.test_function"]
    },
    "group_results": {
      "ModuleName": {
        "tests": {
          "0.TestClass.test_method1": "passed",
          "0.TestClass.test_method2": "failed"
        },
        "summary": {
          "passed": 1,
          "failed": 1,
          "error": 0,
          "skipped": 0,
          "unknown": 0
        }
      }
    }
  }]
}
```

## Workflow

A typical workflow using these scripts:

1. Copy repositories from HDFS (if needed):
   ```bash
   python hdfs_repo_copy.py test.jsonl --output-dir repos/
   ```

2. Run tests on the repositories:
   ```bash
   python run_tests.py --input-file test.jsonl --output-file test_results.jsonl
   ```

3. Group tests and map results:
   ```bash
   python group_and_map_tests.py test.jsonl test_results.jsonl grouped_results.jsonl
   ```

## License

This project is licensed under the MIT License. 