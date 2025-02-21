"""
Disambiguator and Conll builder CLI.

Usage:
    text_to_conll_cli (-i <input> | --input=<input> | -s <string> | --string=<string>)
        (-f <file_type> | --file_type=<file_type>)
        [-b <morphology_db_type> | --morphology_db_type=<morphology_db_type>]
        [-d <disambiguator> | --disambiguator=<disambiguator>]
        [-m <model> | --model=<model>]
    text_to_conll_cli (-h | --help)

Options:
    -i <input> --input=<input>
        A text file or conll file.
    -s <string> --string=<string>
        A string to parse.
    -f <file_type> --file_type=<file_type>
        The type of file passed. Could be 
            conll: conll
            text: raw text
            preprocessed_text: whitespace tokenized text (text will not be cleaned)
            tokenized_tagged: text is already tokenized and POS tagged, in tuple form
            tokenized: text is already tokenized, only parse tokenized input; don't disambiguate to add POS tags or features
    -b <morphology_db_type> --morphology_db_type=<morphology_db_type>
        The morphology database to use; will use camel_tools built-in by default [default: r13]
    -d <disambiguator> --disambiguator=<disambiguator>
        The disambiguation technique used to tokenize the text lines, either 'mle' or 'bert' [default: bert]
    -m <model> --model=<model>
        The name BERT model used to parse (to be placed in the model directory) [default: catib]
    -h --help
        Show this screen.
"""
import os
import time
import logging as lg
import pandas as pd
import networkx as nx
from src.logger import log
from pathlib import Path
from camel_tools.utils.charmap import CharMapper
from src.conll_output import print_to_conll, text_tuples_to_string
from src.data_preparation import get_file_type_params, get_tagset, parse_text
from src.utils.model_downloader import get_model_name
from docopt import docopt
from transformers.utils import logging
from pandas import read_csv
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import torch
def build_dependency_graph(sentence_features_list):
    """Builds a directed dependency graph from parsed sentence tuples."""
    graphs = []  # Store multiple sentence graphs
    for sentence_index, sentence_features in enumerate(sentence_features_list):
        G = nx.DiGraph()
        for word in sentence_features:
            word_id = int(word[0])    # Convert ID from string to integer
            head_id = int(word[6])    # Convert HEAD from string to integer
            word_form = word[1]       # Surface word form
            pos_tag = word[3]         # Part-of-speech tag
            dep_rel = word[7]         # Dependency relation
            lemma = word[2]
            # Add node (word)
            G.add_node(word_id, word=lemma, pos=pos_tag)
            # Add edge (dependency relation)
            if head_id != 0:  # Ignore ROOT (head=0)
                G.add_edge(head_id, word_id, dep_rel=dep_rel)
        graphs.append(G)  # Store the graph for this sentence
    return graphs  # Returns a list of graphs (one per sentence)

def extract_dependency_path(graph, sentence_features, word1, word2):
    """Finds the shortest dependency path between two words in a tuple-based format."""
    lg.info(f"Extracting dependency path between '{word1}' and '{word2}'.")
    # Find node IDs for word1 and word2
    node1 = next((int(word[0]) for word in sentence_features if word[2] == word1), None)
    node2 = next((int(word[0]) for word in sentence_features if word[2] == word2), None)
    if node1 is None or node2 is None:
        lg.warning(f"One or both words ('{word1}', '{word2}') not found in the sentence.")
        return None  # Word not found in the sentence
    try:
        path = nx.shortest_path(graph, source=node1, target=node2)
        path_info = [
            (word[1], word[3], word[7])  # Extract FORM, POS, and DEPREL
            for word in sentence_features if int(word[0]) in path
        ]
        lg.info(f"Found dependency path: {path_info}")
        return path_info
    except nx.NetworkXNoPath:
        lg.warning(f"No dependency path found between '{word1}' and '{word2}'.")
        return None  # No dependency path found

def process_sentence(sentence,file_path, file_type, model_path, model_name, arclean,
                     disambiguator_type, clitic_feats_df, tagset, morphology_db_type, word_pairs):
    """
    Process one sentence: build file_type_params, parse the sentence, build dependency graph,
    and extract dependency paths for word pairs.
    Returns a list of rows (each a list) to be written to the CSV.
    """
    result_rows = []
    s = time.time()
    # Create parameters and parse the sentence
    try:
	    file_type_params = get_file_type_params(
		[sentence],
		'text',
		file_path,
		model_path/model_name,
		arclean,
		disambiguator_type,
		clitic_feats_df,
		tagset,
		morphology_db_type
	    )
	    parsed_text_tuples = parse_text('text', file_type_params)
	    torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print("CUDA OOM error encountered. Clearing cache and skipping sentence.")
        torch.cuda.empty_cache()
        return result_rows  #
    except Exception as exc:
        print("Error parsing a sentence:", exc)
        return result_rows

    e = time.time()
    print("time parsing", (e - s))
    # Process each parsed sentence
    for parsed_sentence in parsed_text_tuples:
        dep_graph = build_dependency_graph([parsed_sentence])[0]
        words_in_sentence = {word[2] for word in parsed_sentence}
        for _, row in word_pairs.iterrows():
            w1, w2 = row["word1"].strip(), row["word2"].strip()
            if w1 in words_in_sentence and w2 in words_in_sentence:
                path = extract_dependency_path(dep_graph, parsed_sentence, w1, w2)
                if path:
                    pos1 = next(word[3] for word in parsed_sentence if word[2] == w1)
                    pos2 = next(word[3] for word in parsed_sentence if word[2] == w2)
                    result_rows.append([w1, w2, pos1, pos2, path, "antonymy"])
    return result_rows

# Set up logging
LOG_FILE = "processing.log"
lg.basicConfig(
    level=lg.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        lg.FileHandler(LOG_FILE),  # Save logs to file
        lg.StreamHandler()         # Print logs to console
    ]
)
lg.info("Script started.")

arguments = docopt(__doc__)
logging.set_verbosity_error()

def get_file_type(file_type):
    if file_type in ['conll', 'text', 'preprocessed_text', 'tokenized_tagged', 'tokenized']:
        return file_type 
    assert False, 'Unknown file type'

def remove_non_arabic_words(text):
    # Remove full words that contain English letters (A-Z, a-z) but keep Arabic, numbers, and punctuation
    text = re.sub(r'\b[A-Za-z]+\b', '', text)  # Remove English words
    text = re.sub(r'[^\u0600-\u06FF0-9\s.,!?؛:\-()]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove unwanted special characters
    return text  # Clean extra spaces

@log
def main():
    print("Main")
    root_dir = Path(__file__).parent
    model_path = root_dir / "models"
    # camel_tools import used to clean text
    arclean = CharMapper.builtin_mapper("arclean")
    #
    ### Get clitic features
    #
    clitic_feats_df = read_csv(root_dir / "data/clitic_feats.csv")
    clitic_feats_df = clitic_feats_df.astype(str).astype(object)  # so ints read are treated as string objects
    #
    ### cli user input ###
    #
    file_path = arguments['--input']
    string_text = arguments['--string']
    file_type = get_file_type(arguments['--file_type'])
    morphology_db_type = arguments['--morphology_db_type']
    disambiguator_type = arguments['--disambiguator']
    parse_model = arguments['--model']
    #
    ### Set up parsing model 
    # (download defaults models, and get correct model name from the models directory)
    #
    model_name = get_model_name(parse_model, model_path=model_path)
    # 
    ### get tagset (depends on model)
    #
    tagset = get_tagset(parse_model)
    #
    ### main code ###
    #
    lines = []
    if string_text is not None:
        lines = [string_text]
    elif file_path is not None:
        with open(file_path, 'r') as f:
            lines = [remove_non_arabic_words(line) for line in f.readlines() if line.strip()]
    # Load word pairs once (before looping through sentences)
    word_pairs = pd.read_csv(
        "/media/randah/0b60ddf9-4b0c-4b37-afaf-93898a41f63a/Dataset/text_pair/hypernyms_pairs_lemma.csv",
        names=["word1", "word2"]
    )
    # Open the CSV file for writing (append mode)
    # Build regex pattern for all pairs.
    # For each pair, include both orders: word1.*word2 and word2.*word1.
    pattern_parts = []
    for _, row in word_pairs.iterrows():
        word1 = re.escape(row["word1"].strip())
        word2 = re.escape(row["word2"].strip())
        pattern_parts.append(f"{word1}.*{word2}")
        pattern_parts.append(f"{word2}.*{word1}")
    # Combine the sub-patterns using the OR operator.
    pattern = "(" + "|".join(pattern_parts) + ")"
    output_file = "/media/randah/0b60ddf9-4b0c-4b37-afaf-93898a41f63a/Dataset/parse_out/hypernyms_dependency_paths_paralel.csv"
    is_new_file = not Path(output_file).exists()  # Check if file exists (needed for headers)
    with open(output_file, "a", encoding="utf-8", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header only if it's a new file
        if is_new_file:
            csv_writer.writerow(["word1", "word2", "pos1", "pos2", "dep_path", "relation"])
        # Process each parsed sentence
        start_time = time.time()
        for line in lines:
            filtered_sentences = [s for s in line.split('.') if re.search(pattern, s)]
            # Parallel processing of each sentence in filtered_sentences
            all_results = []
            with ProcessPoolExecutor() as executor:
                futures = [
                	executor.submit(
			    process_sentence,
			    sentence,
			    file_path,
			    file_type,
			    model_path,
			    model_name,
			    arclean,
			    disambiguator_type,
			    clitic_feats_df,
			    tagset,
			    morphology_db_type,
			    word_pairs
			)
			for sentence in filtered_sentences]
                for future in as_completed(futures):
                    try:
                        all_results.extend(future.result())
                        print("done merging result")
                    except Exception as exc:
                        print("Error processing a sentence:", exc)
            for row in all_results:
                csv_writer.writerow(row)
            csvfile.flush()
            print("done writing result to csv")
            torch.cuda.empty_cache()
        end_time = time.time()
        print("Time for all path saving:", (end_time - start_time) / 3600, "hours")

if __name__ == '__main__':
    main()

