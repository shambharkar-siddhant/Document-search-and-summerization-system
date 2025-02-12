#!/bin/bash
# launch.sh

# Set the OpenAI API key (replace with your actual key or use a secure method)
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

# Function to clean up child processes on exit
cleanup() {
    echo "Shutting down launched processes..."
    # Kill all child processes of this script
    kill 0
    exit 0
}

# Trap common signals and call the cleanup function
trap cleanup SIGINT SIGTERM EXIT

# Launch the API server in the background
echo "Starting api_server.py..."
python3 api_server.py &
PID_API_SERVER=$!

# Launch the processor in the background
echo "Starting processor.py..."
python3 processor.py &
PID_PROCESSOR=$!

echo "Processes launched:"
echo "  API Server PID: ${PID_API_SERVER}"
echo "  Processor PID: ${PID_PROCESSOR}"

# Wait indefinitely so the script doesn't exit immediately.
# The trap will ensure that if this script is terminated, all child processes are killed.
wait
