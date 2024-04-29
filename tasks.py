import gzip
from datetime import datetime, timezone, timedelta
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from sentimentfunction import calculate_sentiment_connection_proportion
from scipy.optimize import curve_fit
# 1
# Limited to only 30 rows because otherwise too slow to test
# Jos testaa, pitää vaa muokata polku twitter dataan että löytyy

start_date = datetime(2019, 10, 1, tzinfo=timezone.utc)
end_date = datetime(2019, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# 2
G = nx.Graph()
shared_hashtags = {}
sentiment_scores = {}
timestamps = []
positive_sentiments = []
negative_sentiments = []

with gzip.open(r'c:\Users\35844\Downloads\TweetsCOV19.tsv.gz', 'rb') as f:
    filtered_rows = []
    number_of_lines = 0

    for line in f:
        fields = line.decode().strip().split('\t')

        timestamp_str = fields[2]
        timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")
        timestamps.append(timestamp)

    
        if start_date <= timestamp <= end_date:
            filtered_rows.append(fields)
            twitter_id = fields[0]
            hashtags = fields[10].split()
            sentiment_str = fields[8]

            # 6 
            positive_sentiment, negative_sentiment = map(int, sentiment_str.split())
            overall_sentiment = positive_sentiment + negative_sentiment
            sentiment_scores[twitter_id] = overall_sentiment
            #print("Overall Sentiment Score:", overall_sentiment)
            positive_sentiments.append(positive_sentiment)
            negative_sentiments.append(negative_sentiment)

            for i, hashtag in enumerate(hashtags):
                if hashtag != "null;":
                    if hashtag in shared_hashtags:
                        for shared_id in shared_hashtags[hashtag]:
                            if shared_id != twitter_id:
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

# 3
largest_component = max(nx.connected_components(G), key=len)

largest_subgraph = G.subgraph(largest_component)

print("Size of the largest component:", len(largest_component))

connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
second_largest_size = len(connected_components[1]) if len(connected_components) > 1 else 0
third_largest_size = len(connected_components[2]) if len(connected_components) > 2 else 0

print("Size of the second largest component:", second_largest_size)
print("Size of the third largest component:", third_largest_size)

avg_path_length = nx.average_shortest_path_length(largest_subgraph)
print("Average path length of the largest component:", avg_path_length)

# 4
degree_centralities = nx.degree_centrality(G)

plt.figure(figsize=(10, 5))
plt.hist(degree_centralities.values(), bins=30, alpha=0.7, color='b', edgecolor='black')
plt.title('Degree Centrality Distribution')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(degree_centralities.values(), bins=30, cumulative=True, alpha=0.7, color='r', edgecolor='black')
plt.title('Cumulative Degree Centrality Distribution')
plt.xlabel('Degree Centrality')
plt.ylabel('Cumulative Frequency')
plt.grid(True)
plt.show()

# 5
degree_centralities_values = list(degree_centralities.values())
results = powerlaw.Fit(degree_centralities_values)

print("Alpha (exponent of the power law):", results.power_law.alpha)
print("Xmin (minimum value):", results.power_law.xmin)

R, p = results.distribution_compare('power_law', 'lognormal')
print("P-value:", p)

# 7
positive_positive_prop, negative_negative_prop, positive_negative_prop, negative_positive_prop = calculate_sentiment_connection_proportion(G, sentiment_scores)

print("Proportion of Positive-Positive Connections:", positive_positive_prop)
print("Proportion of Negative-Negative Connections:", negative_negative_prop)
print("Proportion of Positive-Negative Connections:", positive_negative_prop)
print("Proportion of Negative-Positive Connections:", negative_positive_prop)

# 8
min_timestamp = min(timestamps)
max_timestamp = max(timestamps)
time_interval = timedelta(days=1)  # 1 day interval
num_intervals = (max_timestamp - min_timestamp) // time_interval + 1

counts_per_interval = np.zeros(num_intervals)

# Count positive sentiment tweets for each interval
for timestamp, sentiment in zip(timestamps, positive_sentiments):
    interval_index = (timestamp - min_timestamp) // time_interval
    counts_per_interval[interval_index] += sentiment

time_points = [min_timestamp + i * time_interval for i in range(num_intervals)]
plt.plot(time_points, counts_per_interval)
plt.xlabel('Time')
plt.ylabel('Total Number of Positive Sentiment Tweets')
plt.title('Evolution of Positive Sentiment Tweets over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Fit a parametric distribution to the obtained graph
# For example, you can fit a polynomial function to the data
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

popt, pcov = curve_fit(polynomial_function, np.arange(num_intervals), counts_per_interval)

plt.plot(np.arange(num_intervals), polynomial_function(np.arange(num_intervals), *popt), 'r--')
plt.scatter(np.arange(num_intervals), counts_per_interval)
plt.xlabel('Time Interval')
plt.ylabel('Total Number of Positive Sentiment Tweets')
plt.title('Fitted Polynomial Curve for Positive Sentiment Tweets')
plt.grid(True)
plt.show()

# 9
# Count negative sentiment tweets for each interval
for timestamp, sentiment in zip(timestamps, negative_sentiments):
    interval_index = (timestamp - min_timestamp) // time_interval
    counts_per_interval[interval_index] += sentiment

# Plot the evolution of the total number of negative sentiment tweets over time
time_points = [min_timestamp + i * time_interval for i in range(num_intervals)]
plt.plot(time_points, counts_per_interval)
plt.xlabel('Time')
plt.ylabel('Total Number of Negative Sentiment Tweets')
plt.title('Evolution of Negative Sentiment Tweets over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Fit a parametric distribution to the obtained graph
# For example, you can fit a polynomial function to the data
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the polynomial function to the data
popt, pcov = curve_fit(polynomial_function, np.arange(num_intervals), counts_per_interval)

# Plot the fitted curve
plt.plot(np.arange(num_intervals), polynomial_function(np.arange(num_intervals), *popt), 'r--')
plt.scatter(np.arange(num_intervals), counts_per_interval)
plt.xlabel('Time Interval')
plt.ylabel('Total Number of Negative Sentiment Tweets')
plt.title('Fitted Polynomial Curve for Negative Sentiment Tweets')
plt.grid(True)
plt.show()