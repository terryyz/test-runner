# Test Runner

A simplified version of Repo2Run that implements only the `--run-tests` functionality. This tool reads previously extracted test files from a `test.jsonl` file and runs them using the system Python interpreter.

## Features

- Run tests from previously extracted `test.jsonl` files
- Handle both verbose and non-verbose logging modes
- Run tests in parallel using multiple worker processes
- Timeout for long-running tests
- Generate detailed test results in `test_results.jsonl`

## Requirements

- Python 3.6+
- pytest
- tqdm (for progress bars)
- psutil (for process management)

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

## Usage

```bash
python run_tests.py --output-dir OUTPUT_DIR [--verbose] [--num-workers N] [--timeout SECONDS] [--require-tested-files]
```

### Arguments

- `--output-dir DIR`: Directory containing `test.jsonl` and where to write `test_results.jsonl`
- `--verbose`: Enable verbose logging
- `--num-workers N`: Number of worker processes for parallel processing (default: number of CPU cores)
- `--timeout SECONDS`: Timeout in seconds (default: 1800 - 0.5 hour)
- `--require-tested-files`: Only run test files that have `tested_files` information

## Input Format

The tool reads from a `test.jsonl` file in the specified output directory. This file should contain one JSON object per line, with each object representing a repository and its test files.

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
    }]
  }
}
```

## License

This project is licensed under the MIT License. 