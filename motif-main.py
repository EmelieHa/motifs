"""This cell reads the original tsv file with source text, context and corresponding motifs.
Saves it to clean_topics.tsv which retains two columns: topic, and custom_id
"""

# Import necessary libraries
import os
import csv
import torch
import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Define the file paths
input_file = "[input tsv-file]"  # Input TSV file path
output_file = "clean_topics.tsv"  # Output TSV file path

# Function to process the TSV file and output topics in the desired format
def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
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


"""This cell uses a sentence transformer for English to cluster the motifs.
Parameters can be adjusted accordingly.
Saves to the file clustered_topics.tsv which has three columns: topic, custom_id and topic_id.
Also creates a visualization of the clustering and saves it as topic_visualization.png.
"""


# Load the TSV file
file_path = "clean_topics.tsv"
df = pd.read_csv(file_path, sep='\t')

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Encode the literary motifs into embeddings
motif_embeddings = model.encode(df['topic'].astype(str), show_progress_bar=True)


# Initialize UMAP with adjusted parameters
umap_model = UMAP(n_neighbors=10, n_components=5, metric='cosine', min_dist=0.09)

# Initialize HDBSCAN with adjusted parameters
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True, allow_single_cluster=False)


# Initialize BERTopic with UMAP and HDBSCAN
topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=True)

# Fit the model to the embeddings
topics, probs = topic_model.fit_transform(df['topic'].astype(str), motif_embeddings)

# Save the topics back to the dataframe
df['topic_id'] = topics

# Save the resulting DataFrame into a new TSV file
output_file = "clustered_topics.tsv"
df.to_csv(output_file, sep='\t', index=False)

# Display the topics found
topic_info = topic_model.get_topic_info()
print(topic_info)

# Visualize topics and save the image as an HTML file first, then save it as an image
visualization = topic_model.visualize_topics()

# Save the visualization as an HTML file
html_output_file = "topic_visualization.html"
visualization.write_html(html_output_file)

# Save the visualization as an image (requires kaleido package)
image_output_file = "topic_visualization.png"
visualization.write_image(image_output_file, format="png")

print(f"Visualization saved as {html_output_file} and {image_output_file}")


"""
This cell adds extra information to the tsv file, (in this case time periods for the different titles), and normalizes 
titles to make them more convenient. It also sorts the motifs in numerical order with -1 at the bottom.
Saves the result as sorted_unique_topics.tsv
"""


import pandas as pd

# Load the TXT/TSV file containing novel titles and periods
novel_titles_file = '[input txt/tsv-file]'
novel_titles_df = pd.read_csv(novel_titles_file, sep='\t')

# Precompute a mapping of substring matches
substring_map = {}
for index, row in novel_titles_df.iterrows():
    substring_map.setdefault(row['custom_id'], []).append((row['period'], row['novel_key']))

# Function to assign all matching 'period' and 'novel_key' based on precomputed mapping
def get_period_and_source(custom_id):
    matches = []
    for key, values in substring_map.items():
        if key in custom_id:  # Check if `key` is a substring of `custom_id`
            matches.extend(values)
    return pd.DataFrame(matches, columns=['period', 'novel_key']) if matches else pd.DataFrame(columns=['period', 'novel_key'])

# Load the TSV file with clustered topics
input_file = 'clustered_topics.tsv'
df = pd.read_csv(input_file, sep='\t')

# Step 1: Initialize new columns for 'period' and 'novel_source'
df['period'] = None
df['novel_source'] = None

# Step 2: Process all rows and expand for multiple matches
all_matches = []
for index, row in df.iterrows():
    matches = get_period_and_source(row['custom_id'])
    if not matches.empty:
        for _, match in matches.iterrows():
            new_row = row.copy()
            new_row['period'] = match['period']
            new_row['novel_source'] = match['novel_key']
            all_matches.append(new_row)

# Step 3: Create an expanded DataFrame
if all_matches:
    expanded_df = pd.DataFrame(all_matches)

    # Merge the original DataFrame with the expanded matches
    df = df.merge(
        expanded_df[['custom_id', 'period', 'novel_source']],
        on='custom_id',
        how='left',
        suffixes=('', '_expanded')
    )

    # Fill new columns with expanded data
    df['period'] = df['period_expanded'].combine_first(df['period'])
    df['novel_source'] = df['novel_source_expanded'].combine_first(df['novel_source'])
    df.drop(columns=['period_expanded', 'novel_source_expanded'], inplace=True)

# Step 4: Handle unmatched rows (optional)
unmatched = df[df['period'].isnull()]
if not unmatched.empty:
    unmatched_log_file = 'unmatched_custom_ids.log'
    unmatched.to_csv(unmatched_log_file, sep='\t', index=False)
    print(f"Unmatched custom IDs have been logged to '{unmatched_log_file}'.")

# Step 5: Sort motifs numerically, treating '-1' as the last value
def sort_motif(motif):
    try:
        motif_value = int(motif)
        return float('inf') if motif_value == -1 else motif_value
    except ValueError:
        return float('inf')

df['motif_sort_key'] = df['topic_id'].apply(sort_motif)

# Step 6: Save the updated DataFrame back to the same file
df.sort_values(by='motif_sort_key').drop(columns=['motif_sort_key']).to_csv(input_file, sep='\t', index=False)
print(f"Updated file '{input_file}' has been saved with new columns.")



""" This cell takes the input file, collects all the motif belonging to the same topic, and sends them as a list to 
the language model Llama-3.1-8B-Instruct (change it accordingly). The model is given a prompt that instructs it to
label the cluster as convenient as possible (change prompt accordingly). 
The result is saved as clustered_meta_motifs_with_general_motifs

"""


# Load the TSV file
file_path = "sorted_unique_topics.tsv"
df = pd.read_csv(file_path, sep='\t')



# Model setup for sending clusters to language model
model_id = 'meta-llama/Llama-3.1-8B-Instruct'
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
output_file = "clustered_meta_motifs_with_general_motifs.tsv"
with open(output_file, 'w') as f_out:
    f_out.write("period\tnovel\ttopic\ttopic_id\tgeneral_motif\n")  # Adjust header as per the columns

# Process each unique topic_id (including -1) and generate a general motif
for topic_id in df['topic_id'].unique():
    # Get topics belonging to this topic_id (previously 'motif')
    topics = df[df['topic_id'] == topic_id]['topic'].tolist()
    
    # Send to the model for a general motif
    general_motif = send_topics_to_model(topics)
    
    # Write each row corresponding to this topic_id directly to the file
    with open(output_file, 'a') as f_out:
        for _, row in df[df['topic_id'] == topic_id].iterrows():
            f_out.write(f"{row['period']}\t{row['novel_source']}\t{row['topic']}\t{row['topic_id']}\t{general_motif}\n")

print(f"File with general motifs is being written to: {output_file}")



# Load the TSV file
df = pd.read_csv('clustered_meta_motifs_with_general_motifs.tsv', sep='\t')

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

# Output file path
output_file = 'extracted_general_motifs.tsv'

# Write the sorted data to the output file with the specified column order
df[['period', 'novel', 'topic', 'topic_id', 'general_motif', 'extracted_general_motif']].to_csv(
    output_file, sep='\t', index=False
)

print(f"Extracted and sorted general motifs have been saved to '{output_file}'.")


"""
This file prints the number of text chunks pertaining to each time period, except topic -1. It prints a list of
the biggest motifs that are shared by the highest number of source texts, collected from the 40 biggest motifs overall.

"""


# Load the extracted general motifs TSV file
df_extracted = pd.read_csv('extracted_general_motifs.tsv', sep='\t')

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
    print("Top 40 topics shared by novels:")
    for row in sorted_results[:40]:
        print(f"- {row.extracted_general_motif} (Shared by: {', '.join(row.novel)}) - Total Count: {row.total_count}")

# Analyze the shared topics
analyze_shared_topics(df_extracted)


"""
This cell prints out some information about the results. First it filters out topic -1 as that is not interesting.
It prints out the 40 most common motifs (according to size) plus their total counts.
Then it calculates and prints the relatively most unique three motifs for each novel in the corpus.
"""


# Load the data (assuming a TSV file)
df = pd.read_csv('extracted_general_motifs.tsv', sep='\t')

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




def create_compressed_graph(input_file, output_file="compressed_novel_network_graph.png", top_motifs=40):
    df = pd.read_csv(input_file, sep='\t')

    # Step 1: Select the 40 biggest motifs
    motif_counts = df['topic_id'].value_counts()
    top_motifs = motif_counts.nlargest(top_motifs).index

    # Step 2: Create a matrix where rows are novels and columns are motifs
    novels = df['novel'].unique()

    novel_to_index = {novel: i for i, novel in enumerate(novels)}
    topic_to_index = {topic: i for i, topic in enumerate(top_motifs)}

    # Create a feature matrix based on the top motifs
    feature_matrix = np.zeros((len(novels), len(top_motifs)))
    for _, row in df.iterrows():
        if row['topic_id'] in top_motifs:
            novel_idx = novel_to_index[row['novel']]
            topic_idx = topic_to_index[row['topic_id']]
            feature_matrix[novel_idx, topic_idx] += 1

    # Step 3: Normalize rows (novels) to represent proportions
    normalized_matrix = normalize(feature_matrix, axis=1, norm='l1')

    # Step 4: Dimensionality reduction using UMAP (adjust parameters for compression)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.01, n_components=2, metric='cosine')
    reduced_matrix = reducer.fit_transform(normalized_matrix)

    # Step 5: Create the graph with node positions based on UMAP output
    G = nx.Graph()
    for i, novel in enumerate(novels):
        G.add_node(novel, pos=reduced_matrix[i], time_period=df[df['novel'] == novel]['period'].iloc[0])

    # Step 6: Visualization
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[node]['time_period'] for node in G]

    # Create a colormap for time periods
    unique_periods = list(set(node_colors))
    cmap = plt.cm.get_cmap('tab10', len(unique_periods))
    color_map = {period: cmap(i) for i, period in enumerate(unique_periods)}
    node_colors = [color_map[period] for period in node_colors]

    plt.figure(figsize=(20, 10))

    # Draw the network with nodes only
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_size=1000,  # Slightly larger node size for visibility
        font_size=8,
        node_color=node_colors,  # Node colors based on time period
    )
 

    plt.title("Compressed Vector-Based Network Graph of Novels")
    
    # Save the graph to the specified output file
    plt.savefig(output_file, format='png', bbox_inches='tight')
    plt.show()

# Run the function with the path to your input file
create_compressed_graph("extracted_general_motifs.tsv", "compressed_novel_network_graph.png")


"""
This cell, and the following one, creates bar charts with selected motifs displaying the most fluctuating and most persistent motifs 
compared between the time periods. The motifs are represented by their numbers and can be changed accordingly.
"""

# Load the data
df = pd.read_csv('extracted_general_motifs.tsv', sep='\t')

# Filter out topic_id -1
df_filtered = df[df['topic_id'] != -1]

# Calculate the total number of text chunks per period
total_chunks_per_period = df.groupby('period').size()

# Select specific motif IDs and filter the data
selected_motifs = [3, 4, 10, 19, 21]
df_selected_motifs = df_filtered[df_filtered['topic_id'].isin(selected_motifs)]

# Aggregate counts by period for the selected motifs
motif_counts_by_period = df_selected_motifs.groupby(['period', 'topic_id']).size().unstack(fill_value=0)

# Normalize counts by the total number of chunks per period
relative_motif_counts = motif_counts_by_period.div(total_chunks_per_period, axis=0)

# Extract motif legends for the selected motifs, truncating if necessary
motif_legends = (
    df[df['topic_id'].isin(selected_motifs)]
    .groupby('topic_id')['extracted_general_motif']
    .first()  # Get the first legend for each motif
    .apply(lambda x: x[:50] + '...' if len(x) > 50 else x)  # Truncate if longer than 50 characters
)

# Create the bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Set the positions for the groups and bars
x_positions = np.arange(len(relative_motif_counts.index))  # Positions for time periods
bar_width = 0.15  # Width of each bar
colors = ['blue', 'green', 'orange', 'red', 'purple']  # Colors for motifs

# Plot each motif as a group of bars
for i, motif in enumerate(selected_motifs):
    ax.bar(
        x_positions + i * bar_width,  # Adjust position for each motif
        relative_motif_counts[motif],  # Heights of the bars
        bar_width,
        label=f"{motif}: {motif_legends[motif]}",  # Add motif ID and legend
        color=colors[i % len(colors)]  # Cycle through colors
    )

# Customize the chart
ax.set_title('Most Fluctuating Motifs', fontsize=16)
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Relative Frequency', fontsize=12)
ax.set_xticks(x_positions + (len(selected_motifs) - 1) * bar_width / 2)  # Center the labels
ax.set_xticklabels(relative_motif_counts.index)  # Time period labels
ax.legend(title='Motif ID', fontsize=9, loc='upper right')  # Add legend with smaller font size
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot as PNG
filename = 'most_fluctuating_motifs_bar_chart.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Saved plot as {filename}")

# Show the plot
plt.show()



# Load the data
df = pd.read_csv('extracted_general_motifs.tsv', sep='\t')

# Filter out topic_id -1
df_filtered = df[df['topic_id'] != -1]

# Calculate the total number of text chunks per period
total_chunks_per_period = df.groupby('period').size()

# Select specific motif IDs and filter the data
selected_motifs = [0, 1, 2, 9, 13]
df_selected_motifs = df_filtered[df_filtered['topic_id'].isin(selected_motifs)]

# Aggregate counts by period for the selected motifs
motif_counts_by_period = df_selected_motifs.groupby(['period', 'topic_id']).size().unstack(fill_value=0)

# Normalize counts by the total number of chunks per period
relative_motif_counts = motif_counts_by_period.div(total_chunks_per_period, axis=0)

# Extract motif legends for the selected motifs, truncating if necessary
motif_legends = (
    df[df['topic_id'].isin(selected_motifs)]
    .groupby('topic_id')['extracted_general_motif']
    .first()  # Get the first legend for each motif
    .apply(lambda x: x[:50] + '...' if len(x) > 50 else x)  # Truncate if longer than 50 characters
)

# Create the bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Set the positions for the groups and bars
x_positions = np.arange(len(relative_motif_counts.index))  # Positions for time periods
bar_width = 0.15  # Width of each bar
colors = ['blue', 'green', 'orange', 'red', 'purple']  # Colors for motifs

# Plot each motif as a group of bars
for i, motif in enumerate(selected_motifs):
    ax.bar(
        x_positions + i * bar_width,  # Adjust position for each motif
        relative_motif_counts[motif],  # Heights of the bars
        bar_width,
        label=f"{motif}: {motif_legends[motif]}",  # Add motif ID and legend
        color=colors[i % len(colors)]  # Cycle through colors
    )

# Customize the chart
ax.set_title('Most Persistent Motifs', fontsize=16)
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Relative Frequency', fontsize=12)
ax.set_xticks(x_positions + (len(selected_motifs) - 1) * bar_width / 2)  # Center the labels
ax.set_xticklabels(relative_motif_counts.index)  # Time period labels
ax.legend(title='Motif ID', fontsize=9, loc='upper right')  # Add legend with smaller font size
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot as PNG
filename = 'most_persistent_motifs_bar_chart.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Saved plot as {filename}")

# Show the plot
plt.show()



