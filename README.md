# RAG-Core-Service Project README

## Project Overview

The RAG-Core-Service is a Python application designed to leverage advanced information retrieval techniques using a combination of reranking and BM25. This service is equipped with a user interface built using Streamlit, allowing for easy interaction and efficient querying.

## Key Components

- `main.py`: Contains the Streamlit application code for the user interface. It includes features for reranking and the BM25 scoring option.
- `app.py`: This script is used for running the Chainlit-based UI, tailored for specific workflow enhancements.
- `database.py`: Manages database interactions.
- `util.py`: Contains utility functions used across the project.
- `data/`: Directory where extracted data is stored.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```
## Running the Application
### Streamlit UI
To start the Streamlit UI, execute the following command:

```bash
streamlit run main.py --server.headless=true
```
## Chainlit UI
### For launching the Chainlit UI:
```bash
chainlit run app.py -w
```