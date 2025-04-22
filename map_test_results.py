#!/usr/bin/env python3
"""
Script to map test-runner output (test results) to the grouped tests from group_tests.py.
This takes test results and maps them to the appropriate test groups based on the group field.
"""

import json
import sys
import os
from collections import defaultdict


def parse_test_results(test_runner_output_file):
    """
    Parse the test runner output file to extract test results
    Returns a dictionary mapping test case identifiers to their statuses
    """
    test_output = {}
    with open(test_runner_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line.strip())
            # Merge this line's data into the overall test_output
            for key, value in line_data.items():
                if key not in test_output:
                    test_output[key] = value
                elif isinstance(test_output[key], dict) and isinstance(value, dict):
                    test_output[key].update(value)
                elif isinstance(test_output[key], list) and isinstance(value, list):
                    test_output[key].extend(value)
    
    # Extract the test case status map
    test_case_status_map = {}
    
    if "tests" in test_output and "details" in test_output["tests"]:
        # Extract test results from the test_case_status_map if available
        if "test_case_status_map" in test_output["tests"]:
            test_case_status_map = test_output["tests"]["test_case_status_map"]
        
        # If no status map is available, build one from the detailed results
        if not test_case_status_map:
            for test_details in test_output["tests"]["details"]:
                for test in test_details.get("tests", []):
                    test_name = test.get("name", "")
                    class_name = test.get("classname", "")
                    status = test.get("status", "unknown")
                    
                    # Create a key that can be matched with group test names
                    if class_name:
                        full_name = f"{class_name}.{test_name}"
                        test_case_status_map[full_name] = status
                    else:
                        test_case_status_map[test_name] = status
    
    return test_case_status_map


def map_results_to_groups(grouped_tests_file, test_results, output_file):
    """
    Map test results to grouped tests in the JSONL file
    Updates the JSONL file with test result information for each test group
    """
    output_lines = []
    
    with open(grouped_tests_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Process each test in the "tests" section
                if "tests" in data:
                    # Check if "tests" is a dictionary
                    if isinstance(data["tests"], dict):
                        for test_key, test_info in data["tests"].items():
                            if "group" in test_info:
                                # Initialize results in each group
                                for entity, test_cases in test_info["group"].items():
                                    result_counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "unknown": 0}
                                    results = {}
                                    
                                    # Map test results to each test case in the group
                                    for test_case in test_cases:
                                        # Extract test name parts from the test_case identifier (e.g. "0.TestClass.test_method")
                                        parts = test_case.split(".", 2)
                                        if len(parts) >= 3:
                                            # Try different combinations to match with test result keys
                                            test_class = parts[1]
                                            test_method = parts[2]
                                            possible_keys = [
                                                f"{test_class}.{test_method}",
                                                test_method,
                                                test_case
                                            ]
                                            
                                            # Try to find the test result using possible key formats
                                            test_status = None
                                            for key in possible_keys:
                                                # Direct match in test_case_status_map
                                                if key in test_results:
                                                    test_status = test_results[key]
                                                    break
                                                
                                                # Look for partial matches
                                                for result_key in test_results:
                                                    if key in result_key:
                                                        test_status = test_results[result_key]
                                                        break
                                                if test_status:
                                                    break
                                            
                                            # If still no match, check file-level matches
                                            if not test_status:
                                                module_name = test_class.split('.')[-1]
                                                for result_key in test_results:
                                                    if module_name in result_key:
                                                        test_status = test_results[result_key]
                                                        break
                                            
                                            # Use unknown status if we couldn't find a match
                                            if not test_status:
                                                test_status = "unknown"
                                        else:
                                            test_status = "unknown"
                                        
                                        # Count the result
                                        result_counts[test_status] = result_counts.get(test_status, 0) + 1
                                        results[test_case] = test_status
                                    
                                    # Add the results to the group
                                    test_info["group_results"] = {
                                        entity: {
                                            "tests": results,
                                            "summary": result_counts
                                        }
                                    }
                    
                    # Check if "tests" is a list
                    elif isinstance(data["tests"], list):
                        for i, test_info in enumerate(data["tests"]):
                            if isinstance(test_info, dict) and "group" in test_info:
                                # Initialize results in each group
                                for entity, test_cases in test_info["group"].items():
                                    result_counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "unknown": 0}
                                    results = {}
                                    
                                    # Map test results to each test case in the group
                                    for test_case in test_cases:
                                        # Extract test name parts from the test_case identifier
                                        parts = test_case.split(".", 2)
                                        if len(parts) >= 3:
                                            test_class = parts[1]
                                            test_method = parts[2]
                                            possible_keys = [
                                                f"{test_class}.{test_method}",
                                                test_method,
                                                test_case
                                            ]
                                            
                                            # Try to find the test result using possible key formats
                                            test_status = None
                                            for key in possible_keys:
                                                if key in test_results:
                                                    test_status = test_results[key]
                                                    break
                                                
                                                # Look for partial matches
                                                for result_key in test_results:
                                                    if key in result_key:
                                                        test_status = test_results[result_key]
                                                        break
                                                if test_status:
                                                    break
                                            
                                            # If still no match, check file-level matches
                                            if not test_status:
                                                module_name = test_class.split('.')[-1]
                                                for result_key in test_results:
                                                    if module_name in result_key:
                                                        test_status = test_results[result_key]
                                                        break
                                            
                                            # Use unknown status if we couldn't find a match
                                            if not test_status:
                                                test_status = "unknown"
                                        else:
                                            test_status = "unknown"
                                        
                                        # Count the result
                                        result_counts[test_status] = result_counts.get(test_status, 0) + 1
                                        results[test_case] = test_status
                                    
                                    # Add the results to the group
                                    test_info["group_results"] = {
                                        entity: {
                                            "tests": results,
                                            "summary": result_counts
                                        }
                                    }
                                
                                # Update the test in the data
                                data["tests"][i] = test_info
                
                # Write the updated data to the output file
                output_lines.append(json.dumps(data))
            
            except json.JSONDecodeError:
                # Keep invalid lines unchanged
                output_lines.append(line.strip())
    
    # Write all lines to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')


def map_nested_test_results(test_runner_output):
    """
    Convert nested test_case_status_map to a flattened dictionary for easier lookup
    """
    flattened_results = {}
    
    # First, handle the case where test_case_status_map is directly in the test_runner_output
    if "test_case_status_map" in test_runner_output:
        for module, tests in test_runner_output["test_case_status_map"].items():
            for test_name, status in tests.items():
                flattened_results[f"{module}.{test_name}"] = status
                flattened_results[test_name] = status
    
    # Next, check if it's in the "tests" section
    elif "tests" in test_runner_output and "test_case_status_map" in test_runner_output["tests"]:
        for module, tests in test_runner_output["tests"]["test_case_status_map"].items():
            for test_name, status in tests.items():
                flattened_results[f"{module}.{test_name}"] = status
                flattened_results[test_name] = status
    
    # Finally, try to extract from individual test details
    elif "tests" in test_runner_output and "details" in test_runner_output["tests"]:
        for test_suite in test_runner_output["tests"]["details"]:
            if "tests" in test_suite:
                for test in test_suite["tests"]:
                    name = test.get("name", "")
                    classname = test.get("classname", "")
                    status = test.get("status", "unknown")
                    
                    if classname:
                        flattened_results[f"{classname}.{name}"] = status
                    flattened_results[name] = status
    
    return flattened_results


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} grouped_tests_file test_runner_output output_file")
        sys.exit(1)
    
    grouped_tests_file = sys.argv[1]
    test_runner_output_file = sys.argv[2]
    output_file = sys.argv[3]
    
    print(f"Mapping test results from {test_runner_output_file} to groups in {grouped_tests_file}")
    
    # Load test results from JSONL file
    test_runner_output = {}
    with open(test_runner_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line.strip())
            # Merge this line's data into the overall test_runner_output
            # For simplicity, we'll use the first level keys to merge
            for key, value in line_data.items():
                if key not in test_runner_output:
                    test_runner_output[key] = value
                elif isinstance(test_runner_output[key], dict) and isinstance(value, dict):
                    test_runner_output[key].update(value)
                elif isinstance(test_runner_output[key], list) and isinstance(value, list):
                    test_runner_output[key].extend(value)
    
    # Map test results to a flattened structure for easier lookup
    test_results = map_nested_test_results(test_runner_output)
    
    # Apply the mapping
    map_results_to_groups(grouped_tests_file, test_results, output_file)
    
    print(f"Mapping complete, results written to {output_file}")


if __name__ == "__main__":
    main() 