import csv
import io
import requests


def getImage(number):
    photo_url = f"https://railway-photos.xm9g.net/photos/{number}.webp"

    URLresponse = requests.head(photo_url)
    if URLresponse.status_code == 200:
        url = photo_url
        # image credits:
        csvurl = 'https://railway-photos.xm9g.net/credit.csv'
            
        search_value = number.strip().upper()

        response = requests.get(csvurl)
        response.raise_for_status() 

        csv_content = response.content.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(csv_content))

        photographer = None
        for row in csv_reader:
            if row[0].strip().upper() == search_value:
                photographer = row[1]
                break
        if photographer == None:
            photographer = "Billy Evans"
    else:
        url = None
        photographer = None
        
    return {
        "url": url,
        "photographer": photographer
    }
