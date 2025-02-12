#!/bin/bash
# launch.sh

# Set the OpenAI API key (replace with your actual key or use a secure method)
export OPENAI_API_KEY="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"

# Define a cleanup function to kill only the child processes spawned by this script.
cleanup() {
    # Get a list of child process IDs and kill them.
    CHILD_PIDS=$(jobs -p)
    if [ -n "$CHILD_PIDS" ]; then
        kill $CHILD_PIDS
    fi
    exit 0
}

# Trap termination signals and the EXIT signal to ensure cleanup happens.
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

# Wait indefinitely so that the script remains running and the trap remains active.
wait
