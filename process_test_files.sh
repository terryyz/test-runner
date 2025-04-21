#!/bin/bash

git clone https://github.com/terryyz/test-runner.git
pip install -e test-runner

# Set error handling
set -e
set -o pipefail

# Log file
LOG_FILE="process_test_files.log"
echo "Starting processing at $(date)" > "$LOG_FILE"

# Process each filtered test file
for FILE in /mnt/bn/tiktok-mm-5/aiic/users/terry/filtered_test_tmp_repo_*.jsonl; do
    # Extract the ID from the filename
    FILENAME=$(basename "$FILE")
    ID=$(echo "$FILENAME" | sed -E 's/filtered_test_tmp_repo_(.*)\.jsonl/\1/')
    
    echo "Processing file: $FILENAME with ID: $ID" | tee -a "$LOG_FILE"
    
    # Step 1: Run hdfs_repo_copy.py
    echo "Step 1: Running hdfs_repo_copy.py on $FILENAME" | tee -a "$LOG_FILE"
    python test-runner/hdfs_repo_copy.py "$FILE"
    
    # Step 2: Write test files
    echo "Step 2: Running write_test_files.py" | tee -a "$LOG_FILE"
    python test-runner/write_test_files.py "$FILE" -c "written_test_tmp_repo_$ID.jsonl" -q
    
    # Step 3: Run test-runner
    echo "Step 3: Running test-runner" | tee -a "$LOG_FILE"
    test-runner --input-file "written_test_tmp_repo_$ID.jsonl" --output-file "output_test_tmp_repo_$ID.jsonl" --timeout 900
    
    # Step 4: Copy output file back to Terry's directory
    echo "Step 4: Copying output file to /mnt/bn/tiktok-mm-5/aiic/users/terry/" | tee -a "$LOG_FILE"
    cp "output_test_tmp_repo_$ID.jsonl" /mnt/bn/tiktok-mm-5/aiic/users/terry/
    
    rm -rf tmp_repo_$ID
    
    echo "Finished processing $FILENAME" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "All files processed successfully at $(date)" | tee -a "$LOG_FILE"