#!/bin/bash

# Define directories
INPUT_DIR="../data/input/"
OUTPUT_DIR="../data/output"
RULES_DIR="./SDoH"

JAR_FILE="MedTagger-fit-context.jar"
JAR_URL="https://github.com/OHNLP/AgingNLP/releases/download/v0.1.1/$JAR_FILE"

# Check if the .jar file exists
if [ ! -f "$JAR_FILE" ]; then
  echo "$JAR_FILE not found. Downloading..."
  curl -sL "$JAR_URL" -o "$JAR_FILE"
  if [ $? -ne 0 ] || [ ! -f "$JAR_FILE" ]; then
    echo "Failed to fetch the .jar file. Exiting."
    exit 1
  fi
fi

# Run the model
java -Xms512M -Xmx2000M -jar "$JAR_FILE" "$INPUT_DIR" "$OUTPUT_DIR" "$RULES_DIR"
