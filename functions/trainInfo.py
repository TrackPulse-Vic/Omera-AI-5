import csv
import requests
from io import StringIO

def trainData(search_value):
    csv_url = 'https://railway-photos.xm9g.net/trainsets.csv'
    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        csv_content = StringIO(response.text)
        reader = csv.reader(csv_content)
        header = next(reader)  # Get column names for validation
        print(f"Searching for train: {search_value}")
        for row in reader:
            if len(row) >= 10:  # Ensure row has enough columns
                # Split the first column by dashes and check if search_value is in any part
                train_parts = row[0].split('-')
                if search_value in train_parts:
                    json_data = dict(zip(header, row))  # Use header for keys
                    return json_data
        print(f"Train {search_value} not found")
    except requests.RequestException as e:
        print(f"Error fetching CSV: {e}")
        return None
    return None
print(
trainData('301M'))