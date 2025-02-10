#Import necessary libraries

import transformers
import torch
from peft import PeftModel, PeftConfig
import pandas as pd
import os
import csv
import torch
import ast
import networkx as nx
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import numpy as np
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"



import csv

# Define the file paths
input_file = "themebot_answers.tsv"  # Input TSV file path
output_file = "1_clean_topics.tsv"  # Output TSV file path

# Function to process the TSV file and output topics in the desired format
def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        # Skip the header in the input file
        next(reader, None)  
        
        # Write header for TSV
        writer.writerow(["topic", "custom_id"])
        
        # Process each row in the input file
        for row in reader:
            # The first column is the custom_id; the rest are motifs
            custom_id = row[0]
            motifs = row[1:]
            
            # Write each motif along with its custom_id to the output file
            for motif in motifs:
                if motif.strip():  # Skip empty cells
                    writer.writerow([motif.strip(), custom_id])

# Call the function to process the file
process_file(input_file, output_file)

print(f"Processing complete. The output has been written to {output_file}")




file_path = "1_clean_topics.tsv"
df = pd.read_csv(file_path, sep='\t')

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Encode the literary motifs into embeddings
motif_embeddings = model.encode(df['topic'].astype(str), show_progress_bar=True)


# Initialize UMAP with adjusted parameters
umap_model = UMAP(n_neighbors=5, n_components=5, metric='cosine', min_dist=0.09)

# Initialize HDBSCAN with adjusted parameters
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True, allow_single_cluster=False)


# Initialize BERTopic with UMAP and HDBSCAN
topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=True)

# Fit the model to the embeddings
topics, probs = topic_model.fit_transform(df['topic'].astype(str), motif_embeddings)

# Save the topics back to the dataframe
df['topic_id'] = topics

# Save the resulting DataFrame into a new TSV file
output_file = "2_clustered_topics.tsv"
df.to_csv(output_file, sep='\t', index=False)

# Display the topics found
topic_info = topic_model.get_topic_info()
print(topic_info)

# Visualize topics and save the image as an HTML file first, then save it as an image
visualization = topic_model.visualize_topics()

# Save the visualization as an HTML file
html_output_file = "3_topic_visualization.html"
visualization.write_html(html_output_file)

# Save the visualization as an image (requires kaleido package)
image_output_file = "4_topic_visualization.png"
visualization.write_image(image_output_file, format="png")

print(f"Visualization saved as {html_output_file} and {image_output_file}")




import pandas as pd

# Load the TXT/TSV file containing novel titles and periods
novel_titles_file = 'novel_titles.txt'
novel_titles_df = pd.read_csv(novel_titles_file, sep='\t')

# Precompute a mapping of substring matches
substring_map = {}
for _, row in novel_titles_df.iterrows():
    substring_map.setdefault(row['custom_id'], []).append((row['period'], row['novel_key']))

# Function to assign all matching 'period' and 'novel_key' based on precomputed mapping
def get_period_and_source(custom_id):
    matches = []
    for key, values in substring_map.items():
        if key in custom_id:  # Check if `key` is a substring of `custom_id`
            matches.extend(values)
    return matches

# Load the TSV file with clustered topics
input_file = '2_clustered_topics.tsv'
df = pd.read_csv(input_file, sep='\t')

# Step 1: Initialize a list to collect all expanded rows
all_expanded_rows = []

# Step 2: Expand each row for multiple matches
for _, row in df.iterrows():
    custom_id = row['custom_id']
    matches = get_period_and_source(custom_id)
    
    if matches:
        for period, novel_key in matches:
            expanded_row = row.copy()
            expanded_row['period'] = period
            expanded_row['novel_source'] = novel_key
            all_expanded_rows.append(expanded_row)
    else:
        # Add the row as-is if no matches are found
        row['period'] = None
        row['novel_source'] = None
        all_expanded_rows.append(row)

# Step 3: Create a DataFrame from expanded rows
expanded_df = pd.DataFrame(all_expanded_rows)

# Step 4: Remove duplicate rows, if any
expanded_df.drop_duplicates(inplace=True)

# Save the expanded DataFrame to a new TSV file
output_file = '2_clustered_topics.tsv'
expanded_df.to_csv(output_file, sep='\t', index=False)

print(f"Processing complete. Expanded data saved to {output_file}")





import pandas as pd

# Input and output file paths
input_file = "2_clustered_topics.tsv"
output_file = "5_sorted_topics.tsv"

# Load the TSV file into a DataFrame
df = pd.read_csv(input_file, sep='\t')

# Define the function to sort motifs
def sort_motif(motif):
    try:
        motif_value = int(motif)
        # If the motif is -1, assign it the largest possible value (infinity)
        return float('inf') if motif_value == -1 else motif_value
    except ValueError:
        # If motif is not an integer, return infinity (for non-numeric motifs)
        return float('inf')

# Apply the sort_motif function to the 'topic_id' column and create a new column 'motif_sort_key'
df['motif_sort_key'] = df['topic_id'].apply(sort_motif)

# Sort the DataFrame by the 'motif_sort_key' column
df_sorted = df.sort_values(by='motif_sort_key')

# Optionally, drop the 'motif_sort_key' column if it's no longer needed
df_sorted = df_sorted.drop(columns=['motif_sort_key'])

# Save the sorted DataFrame to a new TSV file
df_sorted.to_csv(output_file, sep='\t', index=False)

print(f"Processing complete. Sorted topics saved to {output_file}")





import pandas as pd
from transformers import pipeline
import torch

# Load the TSV file
file_path = '5_sorted_topics.tsv'
df = pd.read_csv(file_path, sep='\t')

# Model setup for sending clusters to language model
model_id = '/mimer/NOBACKUP/groups/naiss2024-22-361/Eric_Pap/Llama-3.1-8B-Instruct'
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Function to send clustered topics to the language model for a general motif
def send_topics_to_model(topics):
    # Join the topics to send them as one input
    topics_list = "\n".join(topics)
    
    # Prepare the input prompt
    input_messages = [
        {"role": "system", "content": "Read this list of motifs and express them in one comprehensive motif. Do not repeat yourself or include too many conflicting elements."
         "Your response should refer to concrete actions or objects in the motifs."
         "Be as brief as you can."
         "Do not include introductory or explanatory words in your answers, such as 'the overarching motif is...'"
         "Do not express yourself in abstract terms using symbolic or interpretative language."},
        {"role": "user", "content": topics_list},
    ]
    
    # Send input to the model and process the output
    outputs = pipe(
        input_messages,
        max_new_tokens=40,  # Adjust based on response length expectations
        num_beams=10,
        num_return_sequences=1
    )
    
    # Extract and return the model's response
    print(outputs[0]['generated_text'])
    return outputs[0]['generated_text']

# Prepare the output file and write the header
output_file = "6_labelled_motifs.tsv"
with open(output_file, 'w') as f_out:
    # Check if 'period' exists in the DataFrame and prepare header accordingly
    headers = ["period", "novel", "topic", "topic_id", "general_motif"]
    
    # Include other columns from the original DataFrame if they exist
    extra_columns = [col for col in df.columns if col not in headers]
    headers.extend(extra_columns)
    
    # Write the header to the output file
    f_out.write("\t".join(headers) + "\n")

# Process each unique topic_id (including -1) and generate a general motif
for topic_id in df['topic_id'].unique():
    # Get topics belonging to this topic_id (previously 'motif')
    topics = df[df['topic_id'] == topic_id]['topic'].tolist()
    
    # Send to the model for a general motif
    general_motif = send_topics_to_model(topics)
    
    # Now, open the file once and write all the rows for this topic_id
    with open(output_file, 'a') as f_out:
        for _, row in df[df['topic_id'] == topic_id].iterrows():
            # For each row, handle the 'period' column if it doesn't exist
            period = row.get('period', None)  # Default to None if 'period' doesn't exist
            novel_source = row.get('novel_source', None)
            topic = row.get('topic', None)
            topic_id = row.get('topic_id', None)
            
            # Prepare the line to be written to the file
            line = [str(period), str(novel_source), str(topic), str(topic_id), str(general_motif)]
            
            # Add any extra columns
            for col in extra_columns:
                line.append(str(row.get(col, "")))
            
            # Ensure all elements in line are strings before joining
            line = [str(item) if not isinstance(item, list) else ", ".join(map(str, item)) for item in line]
            
            # Write the line to the file
            f_out.write("\t".join(line) + "\n")

print(f"File with general motifs is being written to: {output_file}")





import pandas as pd
import ast

# Load the TSV file
df = pd.read_csv('6_labelled_motifs.tsv', sep='\t')

# Function to extract the general motif from the 'general_motif' column
def extract_general_motif(motif_str):
    try:
        # Parse the string representation of the list
        motifs_list = ast.literal_eval(motif_str)
        # Loop through the parsed list to find the assistant's content
        for item in motifs_list:
            if item['role'] == 'assistant':
                return item['content']  # Return the assistant's content
    except Exception as e:
        print(f"Error parsing motifs: {e}")
        return None

# Extract the general motif for each row
df['extracted_general_motif'] = df['general_motif'].apply(extract_general_motif)

# Convert topic_id to integer for proper numeric sorting (if not already an integer)
df['topic_id'] = pd.to_numeric(df['topic_id'], errors='coerce')

# Count occurrences of each topic_id and sort by the count first, then by topic_id numerically
df['topic_count'] = df.groupby('topic_id')['topic_id'].transform('count')
df = df.sort_values(by=['topic_count', 'topic_id'], ascending=[False, True]).drop(columns=['topic_count'])

# Ensure the 'period' column exists, otherwise create a placeholder column
if 'period' not in df.columns:
    df['period'] = None  # Or you can set a default value if needed

# Output file path
output_file = '7_extracted_motifs.tsv'

# Write the sorted data to the output file with the specified column order
df[['period', 'novel', 'topic', 'topic_id', 'general_motif', 'extracted_general_motif']].to_csv(
    output_file, sep='\t', index=False
)

print(f"Extracted and sorted general motifs have been saved to '{output_file}'.")




# Load the extracted general motifs TSV file
df_extracted = pd.read_csv('7_extracted_motifs.tsv', sep='\t')

# Step 3: Count the number of text chunks for each period
period_counts = df_extracted['period'].value_counts()

# Print the counts for each period
print("Number of analyzed text chunks per period (exclusive topic -1:)")
for period, count in period_counts.items():
    print(f"{period}: {count}")


# Function to analyze shared topics and exclude topics that occur in only one novel
def analyze_shared_topics(df):
    # Count occurrences of each motif and get the top 40 biggest motifs
    top_motifs_counts = df['extracted_general_motif'].value_counts().head(40)
    top_motifs = top_motifs_counts.index

    # Filter the DataFrame to only include these top motifs
    df_top_motifs = df[df['extracted_general_motif'].isin(top_motifs)]

    # Group by extracted_general_motif and count occurrences in 'novel'
    shared_topics = df_top_motifs.groupby('extracted_general_motif')['novel'].apply(lambda x: x.unique().tolist()).reset_index()
    shared_topics['count'] = shared_topics['novel'].apply(len)

    # Add the total count of each motif
    shared_topics['total_count'] = shared_topics['extracted_general_motif'].map(top_motifs_counts)

    # Exclude topics that occur in only one novel
    shared_topics = shared_topics[shared_topics['count'] > 1]

    # Find topics shared by all novels
    total_novels = 15
    all_shared_topics = shared_topics[shared_topics['count'] == total_novels]

    # Initialize a list to hold the results
    results = []

    # Add topics shared by all novels if available
    if not all_shared_topics.empty:
        results.extend(all_shared_topics.head(40).itertuples(index=False))

    # Fill the results to ensure we have 40 topics
    if len(results) < 40:
        remaining_needed = 40 - len(results)
        top_shared_topics = shared_topics.nlargest(remaining_needed + len(results), 'count')
        for row in top_shared_topics.itertuples(index=False):
            if row not in results:
                results.append(row)

     # Sort results first by count and then by total_count
    sorted_results = sorted(results, key=lambda x: (-len(x.novel), -x.total_count))

    # Print the final results (up to 40 topics)
    print("\n\nTop 40 topics shared by novels:")
    for row in sorted_results[:40]:
        print(f"- {row.extracted_general_motif} (Shared by: {', '.join(row.novel)}) - Total Count: {row.total_count}")

# Analyze the shared topics
analyze_shared_topics(df_extracted)





# Load the data (assuming a TSV file)
df = pd.read_csv('7_extracted_motifs.tsv', sep='\t')

# Exclude rows where topic_id is -1
df = df[df['topic_id'] != -1]

# Step 1: Calculate the total number of text chunks per period
period_sizes = df['period'].value_counts()

# Step 2: Show the 3 most unique motifs for each novel and period based on the top 40 biggest motifs
def relative_unique_motifs(df):
    # Filter to include only the top 40 motifs by count
    top_40_topics = df['topic_id'].value_counts().head(40).index
    df_top_40 = df[df['topic_id'].isin(top_40_topics)]

    # Calculate overall topic frequencies across all novels
    overall_topic_freq = df_top_40['topic_id'].value_counts(normalize=True)
    
    # Normalize overall frequencies by period size (relative frequency)
    normalized_topic_freq = df_top_40.groupby(['period', 'topic_id']).size().div(period_sizes, level='period')

    # For each novel, find the 3 most unique motifs by comparing normalized frequencies
    novels = df_top_40['novel'].unique()
    for novel in novels:
        novel_data = df_top_40[df_top_40['novel'] == novel]
        
        # Calculate topic frequencies for this specific novel
        novel_topic_freq = novel_data['topic_id'].value_counts(normalize=True)
        
        # Calculate the relative uniqueness score: novel-specific frequency divided by normalized overall frequency
        uniqueness = novel_topic_freq / normalized_topic_freq.loc[:, novel_topic_freq.index].groupby('topic_id').sum()
        
        # Sort by uniqueness and take the top 3
        unique_topics = uniqueness.nlargest(3)
        
        print(f"Novel: {novel}")
        for topic_id in unique_topics.index:
            motif = novel_data[novel_data['topic_id'] == topic_id]['extracted_general_motif'].values[0]
            print(f" - Unique Topic {topic_id}: {motif} (Relative Uniqueness Score: {unique_topics[topic_id]:.2f})")
        print()
 

# New function to print the 40 biggest motifs with their general motifs and counts
def print_top_40_motifs_with_counts(df):
    # Get the top 40 topics by count
    top_40_topics = df['topic_id'].value_counts().head(40).index
    
    print("Top 40 biggest topics with their general motifs and total counts:")
    for topic_id in top_40_topics:
        topic_data = df[df['topic_id'] == topic_id]
        if not topic_data.empty:
            motif = topic_data['extracted_general_motif'].values[0]
            total_count = len(topic_data)
            print(f"Topic {topic_id}: {motif} (Total Count: {total_count})")
        else:
            print(f"Topic {topic_id}: No motif available")

# Example usage
relative_unique_motifs(df)
print_top_40_motifs_with_counts(df)


import pandas as pd
import numpy as np
import umap
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def create_full_graph(input_file, output_file="full_novel_network_graph.png"):
    # Step 1: Load data
    df = pd.read_csv(input_file, sep='\t')

    # Step 2: Create a matrix where rows are novels and columns are all motifs
    novels = df['novel'].unique()
    motifs = df['topic_id'].unique()

    novel_to_index = {novel: i for i, novel in enumerate(novels)}
    motif_to_index = {motif: i for i, motif in enumerate(motifs)}

    # Initialize the feature matrix
    feature_matrix = np.zeros((len(novels), len(motifs)))
    for _, row in df.iterrows():
        novel_idx = novel_to_index[row['novel']]
        motif_idx = motif_to_index[row['topic_id']]
        feature_matrix[novel_idx, motif_idx] += 1

    # Step 3: Normalize rows (novels) to represent proportions
    normalized_matrix = normalize(feature_matrix, axis=1, norm='l1')

    # Step 4: Perform dimensionality reduction using UMAP
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.001,
        n_components=2,
        metric='cosine',
        random_state=42  # Fixing random state for consistency
    )
    reduced_matrix = reducer.fit_transform(normalized_matrix)

    # Step 5: Calculate pairwise similarity between novels
    pairwise_sim = cosine_similarity(normalized_matrix)

    # Compute similarity statistics, ignoring self-similarity
    np.fill_diagonal(pairwise_sim, 0)  # Set diagonal to 0 to ignore self-similarity
    similarity_df = pd.DataFrame(pairwise_sim, index=novels, columns=novels)

    # Get top most similar pairs and ensure no duplicates (order does not matter)
    most_similar_pairs = []
    seen_pairs = set()  # Track pairs to avoid duplicates

    most_similar = similarity_df.stack().nlargest(10).items()  # Search through top pairs

    for (novel1, novel2), similarity in most_similar:
        if novel1 != novel2:  # Avoid self-similarity
            # Sort the pair to ensure (A, B) == (B, A)
            pair = tuple(sorted((novel1, novel2)))  
            if pair not in seen_pairs:
                # Find shared motifs (optional logic as before)
                shared_motif = df[(df['novel'] == novel1) | (df['novel'] == novel2)]['extracted_general_motif'].mode()
                most_similar_pairs.append((novel1, novel2, similarity, shared_motif[0] if not shared_motif.empty else 'N/A'))
                seen_pairs.add(pair)  # Mark this pair as seen
        if len(most_similar_pairs) == 3:  # Stop after collecting 3 pairs
            break

    # Step 6: Identify least shared motifs (at least 10 occurrences)
    motif_counts = feature_matrix.sum(axis=0)
    least_shared_motifs_indices = np.where(motif_counts >= 10)[0]  # Only motifs with 10+ occurrences
    least_shared_motifs_indices = least_shared_motifs_indices[np.argsort(motif_counts[least_shared_motifs_indices])[:10]]

    # Extract corresponding motifs and their novels
    extracted_least_shared_motifs = []
    for motif_id in least_shared_motifs_indices:
        motif = list(motif_to_index.keys())[motif_id]
        motif_data = df[df['topic_id'] == motif]['extracted_general_motif'].iloc[0]
        novels_with_motif = df[df['topic_id'] == motif]['novel'].unique()
        extracted_least_shared_motifs.append((motif, motif_data, ", ".join(novels_with_motif)))

    # Step 7: Create the graph
    G = nx.Graph()

    # Add nodes with their attributes
    for i, novel in enumerate(novels):
        period = df[df['novel'] == novel]['period'].iloc[0]
        G.add_node(novel, pos=reduced_matrix[i], time_period=period)

    # Add edges based on similarity
    for i in range(len(novels)):
        for j in range(i + 1, len(novels)):
            similarity = pairwise_sim[i, j]
            if similarity > 0.1:  # Threshold to include meaningful edges
                G.add_edge(novels[i], novels[j], weight=similarity * 5)

    # Convert UMAP reduced_matrix to a dictionary for positions
    pos = {novels[i]: reduced_matrix[i] for i in range(len(novels))}
    # Step 8: Draw the network with nodes colored by time period
    plt.figure(figsize=(20, 10))
    node_colors = [G.nodes[n]['time_period'] for n in G.nodes()]
    unique_periods = list(set(node_colors))
    color_map = plt.cm.get_cmap('tab10', len(unique_periods))  # Use a colormap
    color_dict = {period: color_map(i) for i, period in enumerate(unique_periods)}

    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_size=500,
        font_size=8,
        node_color=[color_dict[G.nodes[n]['time_period']] for n in G.nodes()],
        edge_color='gray',
        width=[G[u][v]['weight'] for u, v in G.edges()],  # Thickness based on weights
    )
    plt.title("Full Vector-Based Network Graph of Novels")
    plt.savefig(output_file, format='png', bbox_inches='tight')
    plt.show()

   # Step 9: Output similarity statistics
    print("Novel pairs sorted by similarity scores (most similar at the top):")

    # Create a list of all novel pairs with their similarity scores
    pairwise_sim_list = [
        (novel1, novel2, pairwise_sim[i, j])
        for i, novel1 in enumerate(novels)
        for j, novel2 in enumerate(novels)
        if i < j  # Avoid duplicates and self-pairs
    ]

    # Sort the list by similarity score in descending order
    pairwise_sim_list = sorted(pairwise_sim_list, key=lambda x: x[2], reverse=True)

    # Print the pairs
    for novel1, novel2, similarity in pairwise_sim_list:
        print(f"{novel1} and {novel2}: similarity {similarity:.2f}")


    print("\nLeast shared motifs (Top 10 with at least 10 occurrences):")
    for motif, motif_data, novels in extracted_least_shared_motifs:
        print(f"Motif {motif}: {motif_data}, found in novels: {novels}")

    # Save similarity matrix to a CSV file for further analysis
    similarity_df.to_csv("novel_similarity_matrix.csv")

# Example usage
create_full_graph("7_extracted_motifs.tsv", "full_novel_network_graph.png")





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('7_extracted_motifs.tsv', sep='\t')

# Filter out topic_id -1
df_filtered = df[df['topic_id'] != -1]

# Calculate the total number of text chunks per period
total_chunks_per_period = df.groupby('period').size()

# Aggregate counts by period for all motifs
motif_counts_by_period = df_filtered.groupby(['period', 'topic_id']).size().unstack(fill_value=0)

# Normalize counts by the total number of chunks per period
relative_motif_counts = motif_counts_by_period.div(total_chunks_per_period, axis=0)

# Sum across periods to get the total count per motif
total_motif_counts = motif_counts_by_period.sum(axis=0)

# Select the top 40 motifs with the highest total frequency across all periods
top_40_motifs = total_motif_counts.nlargest(40).index.tolist()

# Filter the relative motif counts to include only the top 40 motifs
relative_motif_counts_top_40 = relative_motif_counts[top_40_motifs]

# Calculate the fluctuation (standard deviation) and persistence (mean) for each motif
fluctuations = relative_motif_counts_top_40.std(axis=0)  # Fluctuation is measured by standard deviation
persistences = relative_motif_counts_top_40.mean(axis=0)  # Persistence is measured by mean

# Select the top 10 fluctuating motifs (top 5 including motif 0)
top_fluctuating_motifs = fluctuations.nlargest(10).index.tolist()
top_persistent_motifs = persistences.nlargest(10).index.tolist()

# Combine fluctuating and persistent motifs into one set to ensure no duplicates
combined_top_motifs = list(set(top_fluctuating_motifs + top_persistent_motifs))

# Fill fluctuating motifs (ensure at least 5)
fluctuating_motifs = top_fluctuating_motifs[:5]
if len(fluctuating_motifs) < 5:
    additional_motifs = [m for m in combined_top_motifs if m not in fluctuating_motifs][:5 - len(fluctuating_motifs)]
    fluctuating_motifs.extend(additional_motifs)

# Fill persistent motifs (ensure at least 5)
persistent_motifs = [m for m in top_persistent_motifs if m not in fluctuating_motifs][:5]
if len(persistent_motifs) < 5:
    additional_motifs = [m for m in combined_top_motifs if m not in (fluctuating_motifs + persistent_motifs)][:5 - len(persistent_motifs)]
    persistent_motifs.extend(additional_motifs)

# Ensure 5 motifs per chart
assert len(fluctuating_motifs) == 5, "Could not create 5 fluctuating motifs."
assert len(persistent_motifs) == 5, "Could not create 5 persistent motifs."


# Extract motif legends for the selected motifs, truncating if necessary
motif_legends = (
    df[df['topic_id'].isin(fluctuating_motifs + persistent_motifs)]
    .groupby('topic_id')['extracted_general_motif']
    .first()  # Get the first legend for each motif
    .apply(lambda x: x[:50] + '...' if len(x) > 50 else x)  # Truncate if longer than 50 characters
)

# Create the bar chart for fluctuating motifs
fig, ax = plt.subplots(figsize=(14, 8))

# Set the positions for the groups and bars
x_positions = np.arange(len(relative_motif_counts.index))  # Positions for time periods
bar_width = 0.15  # Width of each bar
colors = ['blue', 'green', 'orange', 'red', 'purple']  # Colors for motifs

# Plot each fluctuating motif as a group of bars
for i, motif in enumerate(fluctuating_motifs):
    ax.bar(
        x_positions + i * bar_width,  # Adjust position for each motif
        relative_motif_counts[motif],  # Heights of the bars
        bar_width,
        label=f"Fluct {motif}: {motif_legends[motif]}",  # Add motif ID and legend
        color=colors[i % len(colors)]  # Cycle through colors
    )

# Customize the chart
ax.set_title('Most Fluctuating Motifs', fontsize=16)
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Relative Frequency', fontsize=12)
ax.set_xticks(x_positions + (len(fluctuating_motifs) - 1) * bar_width / 2)  # Center the labels
ax.set_xticklabels(relative_motif_counts.index)  # Time period labels
ax.legend(title='Motif ID', fontsize=9, loc='upper right')  # Add legend with smaller font size
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot as PNG
filename = 'most_fluctuating_motifs_bar_chart.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Saved plot as {filename}")

# Fluctuation graph customization remains unchanged...

# Show and save the plot
plt.show()
filename = 'most_fluctuating_motifs_bar_chart.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Saved plot as {filename}")

# Print fluctuation scores for the top 5 fluctuating motifs
print("\nFluctuation Scores for Most Fluctuating Motifs:")
for motif in fluctuating_motifs:
    print(f"Motif {motif}: Std. Dev. = {fluctuations[motif]:.4f}")


# Show the plot
plt.show()

# Persistence graph customization remains unchanged...

# Show and save the plot
plt.show()
filename = 'most_persistent_motifs_bar_chart.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Saved plot as {filename}")

# Print persistence scores for the top 5 persistent motifs
print("\nPersistence Scores for Most Persistent Motifs:")
for motif in persistent_motifs:
    print(f"Motif {motif}: Mean = {persistences[motif]:.4f}")


# Create the bar chart for persistent motifs
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each persistent motif as a group of bars
for i, motif in enumerate(persistent_motifs):
    ax.bar(
        x_positions + i * bar_width,  # Adjust position for each motif
        relative_motif_counts[motif],  # Heights of the bars
        bar_width,
        label=f"Persist {motif}: {motif_legends[motif]}",  # Add motif ID and legend
        color=colors[i % len(colors)]  # Cycle through colors
    )

# Customize the chart
ax.set_title('Most Persistent Motifs', fontsize=16)
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Relative Frequency', fontsize=12)
ax.set_xticks(x_positions + (len(persistent_motifs) - 1) * bar_width / 2)  # Center the labels
ax.set_xticklabels(relative_motif_counts.index)  # Time period labels
ax.legend(title='Motif ID', fontsize=9, loc='upper right')  # Add legend with smaller font size
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot as PNG
filename = 'most_persistent_motifs_bar_chart.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Saved plot as {filename}")

# Show the plot
plt.show()



import pandas as pd

# Load the data
df = pd.read_csv('7_extracted_motifs.tsv', sep='\t')

# Get unique motifs and their respective counts from the 'extracted_general_motif' column
motif_counts = df['extracted_general_motif'].value_counts()

# Pretty print unique motifs with their total counts
print("Unique Motifs and their Counts from 'extracted_general_motif' column:")
for idx, (motif, count) in enumerate(motif_counts.items(), start=1):
    print(f"{idx}. {motif} - {count} occurrences")



