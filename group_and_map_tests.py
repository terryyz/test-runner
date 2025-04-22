#!/usr/bin/env python3
"""
Script that combines:
1. Adding a "group" key to tests in a JSONL file, mapping tested modules to test names
2. Mapping test results from test runner output to these grouped tests
"""

import json
import sys
import os
import ast
from collections import defaultdict
from tqdm import tqdm
import multiprocessing
from functools import partial
import time

# ==========================================================
# Code copied from group_tests.py for grouping tests
# ==========================================================

def extract_test_cases(test_content):
    """
    Extract test class and method names from test content using AST
    Returns a list of tuples (test_class_name, test_method_name)
    """
    test_cases = []
    
    try:
        # Parse the test content with AST
        tree = ast.parse(test_content)
        
        # Find test classes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # Look for methods starting with 'test_'
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                        method_name = item.name
                        test_cases.append((class_name, method_name))
        
        # If no test classes found, look for standalone test functions
        if not test_cases:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    method_name = node.name
                    test_cases.append(('TestModule', method_name))  # Use a dummy class name
    
    except SyntaxError:
        pass
    
    return test_cases

def parse_target_file(file_path):
    """
    Parse the target file to extract function, method, and class names
    """
    if not os.path.exists(file_path):
        return [], [], {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
            
            functions = []
            classes = []
            methods = defaultdict(list)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's a method or a function
                    if hasattr(node, 'parent_class'):
                        methods[node.parent_class].append(node.name)
                    else:
                        functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    # Mark methods with their parent class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            item.parent_class = node.name
                            methods[node.name].append(item.name)
            
            return functions, classes, methods
        except SyntaxError:
            return [], [], {}
    except Exception:
        return [], [], {}

def extract_imports(test_content):
    """
    Extract imported modules and functions/classes from the test content using AST
    """
    imported_items = []
    
    try:
        tree = ast.parse(test_content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imported_items.append((None, name.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for name in node.names:
                    if name.name == '*':
                        imported_items.append((module, '*'))
                    else:
                        imported_items.append((module, name.name))
    
    except SyntaxError:
        pass
    
    return imported_items

def analyze_test_case(test_class, test_method, test_content, target_file, target_entities):
    """
    Analyze a test case to determine which entities from the target file it tests using AST
    """
    functions, classes, methods = target_entities
    
    tested_entities = []
    
    try:
        # Parse the test content
        tree = ast.parse(test_content)
        
        # Find the method definition
        method_node = None
        class_node = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == test_class:
                class_node = node
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == test_method:
                        method_node = item
                        break
                if method_node:
                    break
        
        # If not found in a class, look for standalone function
        if not method_node:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == test_method:
                    method_node = node
                    break
        
        if not method_node:
            return []
        
        # Extract references in the method body
        references = set()
        for node in ast.walk(method_node):
            if isinstance(node, ast.Name):
                references.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                references.add(f"{node.value.id}.{node.attr}")
        
        # Check if any function from the target file is directly referenced
        for func in functions:
            if func in references:
                tested_entities.append(func)
        
        # Check for class references
        for cls in classes:
            if cls in references:
                tested_entities.append(cls)
                # Also check for method calls on this class
                for method in methods[cls]:
                    if f"{cls}.{method}" in references:
                        tested_entities.append(f"{cls}.{method}")
        
        # If no direct references found, use method name heuristics
        if not tested_entities:
            # Remove 'test_' prefix to guess the function name
            if test_method.startswith('test_'):
                possible_func = test_method[5:]
                if possible_func in functions:
                    tested_entities.append(possible_func)
            
            # Check for class name in test class name
            for cls in classes:
                if cls in test_class:
                    tested_entities.append(cls)
        
        # If still no matches, use imported module analysis
        if not tested_entities:
            # Get the basename of the target file without extension
            target_basename = os.path.basename(target_file)
            if '.' in target_basename:
                target_basename = target_basename.rsplit('.', 1)[0]
            
            imports = extract_imports(test_content)
            for module, item in imports:
                # Check if the import might be related to the target file
                if module and (target_basename in module or module.endswith(target_basename)):
                    if item == '*':
                        # If we imported everything, all functions and classes are candidates
                        tested_entities.extend(functions)
                        tested_entities.extend(classes)
                    elif item in functions:
                        tested_entities.append(item)
                    elif item in classes:
                        tested_entities.append(item)
    
    except SyntaxError:
        pass
    
    return tested_entities

# ==========================================================
# Code for mapping test results to grouped tests
# ==========================================================

def map_test_results(data, test_results):
    """
    Map test results to the grouped tests in the data structure
    """
    # Process each test in the "tests" section
    if "tests" in data:
        # Check if "tests" is a dictionary
        if isinstance(data["tests"], dict):
            for test_key, test_info in data["tests"].items():
                if "group" in test_info:
                    # Initialize group_results if not present
                    if "group_results" not in test_info:
                        test_info["group_results"] = {}
                    
                    # Process each group
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
                                    # Direct match in test_results
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
                        
                        # Add the results to the group_results
                        test_info["group_results"][entity] = {
                            "tests": results,
                            "summary": result_counts
                        }
                    
                    # Update the test in the data dictionary
                    data["tests"][test_key] = test_info
        
        # Handle list format
        elif isinstance(data["tests"], list):
            for i, test_info in enumerate(data["tests"]):
                if isinstance(test_info, dict) and "group" in test_info:
                    # Initialize group_results if not present
                    if "group_results" not in test_info:
                        test_info["group_results"] = {}
                    
                    # Process each group
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
                                    # Direct match in test_results
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
                        
                        # Add the results to the group_results
                        test_info["group_results"][entity] = {
                            "tests": results,
                            "summary": result_counts
                        }
                    
                    # Update the test in the data list
                    data["tests"][i] = test_info
    
    return data

def extract_test_results_map(test_runner_output):
    """
    Extract test results from the test runner output
    Returns a flattened dictionary mapping test names to their status
    """
    flattened_results = {}
    
    # Handle nested test_case_status_map format
    if "tests" in test_runner_output and "test_case_status_map" in test_runner_output["tests"]:
        for module, tests in test_runner_output["tests"]["test_case_status_map"].items():
            for test_name, status in tests.items():
                flattened_results[f"{module}.{test_name}"] = status
                flattened_results[test_name] = status
    
    # Handle detailed test result format
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

def process_line_with_results(line, test_results, repository_base_path=""):
    """
    Process a single line from the JSONL file and add test result mapping
    Returns the processed line as a string
    """
    try:
        data = json.loads(line.strip())
        repository_path = data.get("repository", repository_base_path)
        
        # First, add the group information (from the original group_tests.py logic)
        if "tests" in data:
            # Check if "tests" is a dictionary
            if isinstance(data["tests"], dict):
                # Iterate over the tests dictionary items
                for test_key, test_info in data["tests"].items():
                    # Initialize group if not present
                    if "group" not in test_info:
                        test_info["group"] = {}
                    
                    # Process the test content and add tested entities to groups
                    if "content" in test_info:
                        test_content_list = test_info["content"]
                        
                        # The test_key is the tested file path, we may need to concatenate with repository path
                        tested_file = repository_path + "/" + test_key
                        
                        # Collect all test cases from all content strings
                        all_test_cases = []
                        for j, content in enumerate(test_content_list):
                            test_cases = extract_test_cases(content)
                            for test_class, test_method in test_cases:
                                all_test_cases.append((j, test_class, test_method, content))
                        
                        # Process the target file (which is the test_key)
                        
                        # Extract entities from the target file
                        target_entities = parse_target_file(tested_file)
                        
                        # Collect all test cases and their names for potential fallback grouping
                        all_test_names = []
                        
                        # Analyze each test case
                        for j, test_class, test_method, content in all_test_cases:
                            tested_entities = analyze_test_case(
                                test_class, test_method, content, tested_file, target_entities
                            )
                            
                            # Store test name for potential fallback grouping
                            test_name = f"{j}.{test_class}.{test_method}"
                            all_test_names.append(test_name)
                            
                            # Add the test case to the group dictionary
                            for entity in tested_entities:
                                if entity not in test_info["group"]:
                                    test_info["group"][entity] = []
                                if test_name not in test_info["group"][entity]:
                                    test_info["group"][entity].append(test_name)
                        
                        # If no entities were found for any test case, create a fallback group using the tested file
                        if not test_info["group"] and all_test_names:
                            # Use just the basename to avoid path issues
                            file_basename = os.path.basename(tested_file)
                            test_info["group"][file_basename] = all_test_names
                    
                    # Update the test in the data dictionary
                    data["tests"][test_key] = test_info
            elif isinstance(data["tests"], list):
                # Iterate over the tests list items
                for i, test_info in enumerate(data["tests"]):
                    # Initialize group if not present
                    if isinstance(test_info, dict):
                        if "group" not in test_info:
                            test_info["group"] = {}
                        
                        # Process the test content and add tested entities to groups
                        if "content" in test_info and "file" in test_info:
                            test_content_list = test_info["content"]
                            tested_file = test_info["file"]  # The "file" field contains the tested file
                            
                            # Collect all test cases from all content strings
                            all_test_cases = []
                            for j, content in enumerate(test_content_list):
                                test_cases = extract_test_cases(content)
                                for test_class, test_method in test_cases:
                                    all_test_cases.append((j, test_class, test_method, content))
                            
                            # Process the target file
                            
                            # Extract entities from the target file
                            target_entities = parse_target_file(tested_file)
                            
                            # Collect all test cases and their names for potential fallback grouping
                            all_test_names = []
                            
                            # Analyze each test case
                            for j, test_class, test_method, content in all_test_cases:
                                tested_entities = analyze_test_case(
                                    test_class, test_method, content, tested_file, target_entities
                                )
                                
                                # Store test name for potential fallback grouping
                                test_name = f"{j}.{test_class}.{test_method}"
                                all_test_names.append(test_name)
                                
                                # Add the test case to the group dictionary
                                for entity in tested_entities:
                                    if entity not in test_info["group"]:
                                        test_info["group"][entity] = []
                                    if test_name not in test_info["group"][entity]:
                                        test_info["group"][entity].append(test_name)
                            
                            # If no entities were found for any test case, create a fallback group using the tested file
                            if not test_info["group"] and all_test_names:
                                # Use just the basename to avoid path issues
                                file_basename = os.path.basename(tested_file)
                                test_info["group"][file_basename] = all_test_names
                        elif "content" in test_info and "path" in test_info:
                            test_content_list = test_info["content"]
                            tested_file = test_info["path"]  # The "path" field contains the tested file
                            
                            # Collect all test cases from all content strings
                            all_test_cases = []
                            for j, content in enumerate(test_content_list):
                                test_cases = extract_test_cases(content)
                                for test_class, test_method in test_cases:
                                    all_test_cases.append((j, test_class, test_method, content))
                            
                            # Process the target file
                            
                            # Extract entities from the target file
                            target_entities = parse_target_file(tested_file)
                            
                            # Collect all test cases and their names for potential fallback grouping
                            all_test_names = []
                            
                            # Analyze each test case
                            for j, test_class, test_method, content in all_test_cases:
                                tested_entities = analyze_test_case(
                                    test_class, test_method, content, tested_file, target_entities
                                )
                                
                                # Store test name for potential fallback grouping
                                test_name = f"{j}.{test_class}.{test_method}"
                                all_test_names.append(test_name)
                                
                                # Add the test case to the group dictionary
                                for entity in tested_entities:
                                    if entity not in test_info["group"]:
                                        test_info["group"][entity] = []
                                    if test_name not in test_info["group"][entity]:
                                        test_info["group"][entity].append(test_name)
                            
                            # If no entities were found for any test case, create a fallback group using the tested file
                            if not test_info["group"] and all_test_names:
                                # Use just the basename to avoid path issues
                                file_basename = os.path.basename(tested_file)
                                test_info["group"][file_basename] = all_test_names
                        
                        # Update the test in the data dictionary
                        data["tests"][i] = test_info
        
        # Now, map test results to the groups
        data = map_test_results(data, test_results)
        
        # Return the processed line
        return json.dumps(data)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None

def process_chunk_with_results(chunk, chunk_id, total_chunks, temp_dir, test_results, repository_base_path=""):
    """
    Process a chunk of lines from the JSONL file with test results
    """
    output_file = f"{temp_dir}/chunk_{chunk_id}.jsonl"
    processed_tests = 0
    
    # Create temp dir if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Create a progress bar for this chunk
    desc = f"Chunk {chunk_id+1}/{total_chunks}"
    with tqdm(total=len(chunk), desc=desc, unit="lines", position=chunk_id) as pbar:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in chunk:
                processed_line = process_line_with_results(line, test_results, repository_base_path)
                if processed_line:
                    f.write(processed_line + '\n')
                    processed_tests += 1
                
                pbar.update(1)
                pbar.set_postfix(processed_tests=processed_tests)
    
    return output_file

def process_jsonl_file_with_results(input_file, test_runner_output_file, output_file, num_processes=None):
    """
    Process a JSONL file in parallel, adding a "group" key to each test and mapping test results
    """
    # Determine the number of processes to use
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"Using {num_processes} processes for parallel processing")
    
    # Make sure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Remove output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Create a temporary directory for intermediate files
    temp_dir = f"{output_dir if output_dir else '.'}/temp_{int(time.time())}"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Load test runner output as JSONL file
    print(f"Loading test results from {test_runner_output_file}")
    
    # Process test results line by line for JSONL format
    test_results = {}
    with open(test_runner_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                test_runner_output = json.loads(line.strip())
                # Extract and merge test results from this line
                line_results = extract_test_results_map(test_runner_output)
                test_results.update(line_results)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from line in test runner output: {e}")
                continue
    
    print(f"Extracted {len(test_results)} test results")
    
    # Count total lines for preprocessing
    print("Counting lines in input file...")
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8', errors='replace'))
    print(f"Total lines: {total_lines}")
    
    # Read all lines to memory for distribution to workers
    print("Reading input file...")
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # Distribute lines into chunks for parallel processing
    chunk_size = max(1, len(lines) // num_processes)
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    total_chunks = len(chunks)
    print(f"Dividing data into {total_chunks} chunks")
    
    # Try to get repository base path from the first valid line
    repository_base_path = ""
    for line in lines[:10]:  # Check first 10 lines
        try:
            data = json.loads(line.strip())
            if "repository" in data:
                repository_base_path = data["repository"]
                break
        except:
            continue
    
    # Process chunks in parallel
    print(f"Processing {total_chunks} chunks in parallel...")
    
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Define the worker function with fixed arguments
    worker_func = partial(process_chunk_with_results, 
                          total_chunks=total_chunks, 
                          temp_dir=temp_dir,
                          test_results=test_results,
                          repository_base_path=repository_base_path)
    
    # Submit all chunks for processing
    chunk_futures = []
    for i, chunk in enumerate(chunks):
        chunk_futures.append((i, pool.apply_async(worker_func, (chunk, i))))
    
    # Wait for all chunks to complete and collect results
    temp_files = []
    for i, future in chunk_futures:
        try:
            temp_file = future.get()
            temp_files.append(temp_file)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
    
    # Close and join the pool
    pool.close()
    pool.join()
    
    # Merge all temporary files into the output file
    print("Merging results...")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    print(f"Parallel processing complete. Results written to {output_file}")

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} input_file test_runner_output output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    test_runner_output_file = sys.argv[2]
    output_file = sys.argv[3]
    
    print(f"Processing {input_file} with test results from {test_runner_output_file} -> {output_file}")
    start_time = time.time()
    process_jsonl_file_with_results(input_file, test_runner_output_file, output_file)
    end_time = time.time()
    print(f"Done processing! Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Set up multiprocessing to spawn rather than fork on Unix-based systems
    # This helps avoid any potential issues with forking
    multiprocessing.set_start_method('spawn', force=True)
    main() 