import csv
import gzip
from datetime import datetime, timezone


# 1. Use the timestamp attribute to restrict the collection to Twitter data in the period October 2019-December 2019
# Limited to only 30 rows because otherwise too slow to test
# Jos testaa, pitää vaa muokata polku twitter dataan että löytyy

# Define the start and end dates for the period of interest in UTC timezone
start_date = datetime(2019, 10, 1, tzinfo=timezone.utc)
end_date = datetime(2019, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Open the gzip-compressed TSV file in binary mode
with gzip.open(r'c:\Users\35844\Downloads\TweetsCOV19.tsv.gz', 'rb') as f:
    # Initialize an empty list to store the filtered rows
    filtered_rows = []

    # Initialize a counter to track the number of lines read
    number_of_lines = 0

    # Read the file line by line
    for line in f:
        # Decode the line from bytes to string and split it into fields
        fields = line.decode().strip().split('\t')

        # Extract the timestamp from the fields (assuming it's the third field)
        timestamp_str = fields[2]

        # Convert the timestamp string to a datetime object
        timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")

        # Check if the timestamp falls within the specified date range
        if start_date <= timestamp <= end_date:
            # Append the fields to the list of filtered rows
            filtered_rows.append(fields)

        # Increment the line counter
        number_of_lines += 1

        # If we have read 20 lines, break the loop
        if number_of_lines >= 27:
            break

# Print the filtered rows
for row in filtered_rows:
    print(row)
