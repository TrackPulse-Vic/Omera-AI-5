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
                    image_url = getImage(search_value)
                    if image_url:
                        json_data['image_url'] = image_url
                    return json_data
        print(f"Train {search_value} not found")
    except requests.RequestException as e:
        print(f"Error fetching CSV: {e}")
        return None
    return None

def getImage(train_number):
    image_mapping = {
        "37417": "https://railway-photos.xm9g.net/37417.jpg",
        "43096": "https://railway-photos.xm9g.net/43096.jpg"
    }
    return image_mapping.get(train_number)