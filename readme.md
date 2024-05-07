# HMM-Based POS Tagger

This project implements a Part-of-Speech (POS) tagger using a Hidden Markov Model (HMM). It includes code for training the HMM on a dataset, creating a vocabulary, and using two decoding algorithms (Greedy and Viterbi) for POS tagging.

## Getting Started

These instructions will guide you on how to set up and run the POS tagger on your local machine.

### Prerequisites

Before running the script, ensure you have Python installed on your machine. The script is compatible with Python 3.x. Additionally, the following Python libraries are required:

- `json`: For handling JSON files.
- `collections`: Specifically, the `Counter` class for counting word occurrences.

### Preparing the Data

Your dataset should be divided into three files: `train`, `dev`, and `test`. These files should be placed in a directory named `data`. Ensure that the data is formatted correctly as per the script requirements.

### Running the Script

To run the script, follow these steps:

1. Navigate to the directory where you have the script.
2. Run the script using Python:

   python3 homework.py


The script will process the data, train the HMM, and output the POS tags in the files greedy.out and viterbi.out.

Output
The output from the script will be two files:

greedy.out: Contains the POS tags predicted using the Greedy decoding algorithm.
viterbi.out: Contains the POS tags predicted using the Viterbi decoding algorithm.
Each line in these files will have a word, its corresponding POS tag, and an index.


