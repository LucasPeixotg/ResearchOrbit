#!/bin/bash

# Start Ollama in the background.
ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieving model..."
echo "PULLING gemma2:2b..."
ollama pull gemma2:2b
echo "PULLING qwen2.5:0.5b..."
ollama pull qwen2.5:0.5b
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid