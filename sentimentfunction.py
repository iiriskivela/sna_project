# Function to calculate the proportion of sentiment connections
def calculate_sentiment_connection_proportion(graph, sentiment_scores):
    positive_positive_count = 0
    negative_negative_count = 0
    positive_negative_count = 0
    negative_positive_count = 0

    for u, v in graph.edges():
        u_sentiment = sentiment_scores.get(u, 0)
        v_sentiment = sentiment_scores.get(v, 0)

        if u_sentiment > 0 and v_sentiment > 0:
            positive_positive_count += 1
        elif u_sentiment < 0 and v_sentiment < 0:
            negative_negative_count += 1
        else:
            if u_sentiment > 0 and v_sentiment < 0:
                positive_negative_count += 1
            elif u_sentiment < 0 and v_sentiment > 0:
                negative_positive_count += 1

    total_connections = graph.number_of_edges()
    positive_positive_proportion = positive_positive_count / total_connections
    negative_negative_proportion = negative_negative_count / total_connections
    positive_negative_proportion = positive_negative_count / total_connections
    negative_positive_proportion = negative_positive_count / total_connections

    return (positive_positive_proportion, negative_negative_proportion, positive_negative_proportion, negative_positive_proportion)
