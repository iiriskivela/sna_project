import csv
import gzip
from datetime import datetime, timezone
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# 1. Use the timestamp attribute to restrict the collection to Twitter data in the period October 2019-December 2019
# Limited to only 30 rows because otherwise too slow to test
# Jos testaa, pitää vaa muokata polku twitter dataan että löytyy

start_date = datetime(2019, 10, 1, tzinfo=timezone.utc)
end_date = datetime(2019, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

G = nx.Graph()
shared_hashtags = {}

with gzip.open(r'c:\Users\35844\Downloads\TweetsCOV19.tsv.gz', 'rb') as f:
    filtered_rows = []
    number_of_lines = 0

    for line in f:
        fields = line.decode().strip().split('\t')

        timestamp_str = fields[2]

        timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")

        if start_date <= timestamp <= end_date:
            filtered_rows.append(fields)
            twitter_id = fields[0]
            hashtags = fields[10].split()

            #print("Twitter ID:", twitter_id)
            #print("Hashtags:")
            #for hashtag in hashtags:
            #    print("- ", hashtag)

            for i, hashtag in enumerate(hashtags):
                if hashtag != "null;":
                    if hashtag in shared_hashtags:
                        for shared_id in shared_hashtags[hashtag]:
                            if shared_id != twitter_id:
                                # Add an edge between the current Twitter ID and the shared ID
                                G.add_edge(twitter_id, shared_id)
                                #print(f"Twitter IDs {twitter_id} and {shared_id} share hashtag: {hashtag}")
                        shared_hashtags[hashtag].append(twitter_id)
                    else:
                        shared_hashtags[hashtag] = [twitter_id]

        number_of_lines += 1

        # If we have read x lines, break the loop
        if number_of_lines >= 1000:
            break

#for row in filtered_rows:
#    print(row)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print("Size of the graph:")
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)

# save the adjancency matrix
adj_matrix = nx.to_numpy_array(G)

# Find the largest connected component in the graph
largest_component = max(nx.connected_components(G), key=len)

# Create a subgraph of the largest component
largest_subgraph = G.subgraph(largest_component)

# Print the size of the largest connected component
print("Size of the largest component:", len(largest_component))

# Find the sizes of the second and third largest components, if they exist
connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
second_largest_size = len(connected_components[1]) if len(connected_components) > 1 else 0
third_largest_size = len(connected_components[2]) if len(connected_components) > 2 else 0

print("Size of the second largest component:", second_largest_size)
print("Size of the third largest component:", third_largest_size)

# Calculate the average path length of the largest connected component
avg_path_length = nx.average_shortest_path_length(largest_subgraph)
print("Average path length of the largest component:", avg_path_length)

# Calculate degree centrality for each node
degree_centralities = nx.degree_centrality(G)

# Plot degree centrality distribution
plt.figure(figsize=(10, 5))
plt.hist(degree_centralities.values(), bins=30, alpha=0.7, color='b', edgecolor='black')
plt.title('Degree Centrality Distribution')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot cumulative degree centrality distribution
plt.figure(figsize=(10, 5))
plt.hist(degree_centralities.values(), bins=30, cumulative=True, alpha=0.7, color='r', edgecolor='black')
plt.title('Cumulative Degree Centrality Distribution')
plt.xlabel('Degree Centrality')
plt.ylabel('Cumulative Frequency')
plt.grid(True)
plt.show()