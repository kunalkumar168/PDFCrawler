#!/bin/bash

# prepare a virtual environment for this tool
python3 -m virtualenv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# export OPENAI_API_KEY="your_api_key"
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# Remind user to source the virtual environment before use
echo "Setup complete. To activate the virtual environment, run:"
echo "source venv/bin/activate"