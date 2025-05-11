# Code taken from TrackPulse Vic

from hashlib import sha1
import hmac
from dotenv import dotenv_values
import os

def getUrl(request):
    # ENV READING
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env_path = os.path.join(parent_folder, '.env')
    config = dotenv_values(env_path)

    
    devId = config['DEV_ID']
    key = bytes(config['KEY'], "utf-8")
    request = request + ('&' if ('?' in request) else '?')
    raw = request + 'devid={0}'.format(devId)
    
    # Encode the raw string
    raw_encoded = raw.encode('utf-8')
    
    hashed = hmac.new(key, raw_encoded, sha1)
    signature = hashed.hexdigest()
    return 'http://timetableapi.ptv.vic.gov.au' + raw + '&signature={1}'.format(devId, signature)
