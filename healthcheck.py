def pinghealthcheck(service:str='bot'):
    import os
    from dotenv import load_dotenv
    import requests

    # Load environment variables from .env file
    load_dotenv()
    if service == 'backend':
        uuid = os.getenv('BACKEND_HEALTHCHECK_UUID')
    else:
        uuid = os.getenv('HEALTHCHECK_UUID')
    print(f"Health check UUID: {uuid}")

    url = f'https://hc-ping.com/{uuid}'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Health check successful.")
        else:
            print(f"Health check failed with status code: {response.status_code}, check the UUID is set correctly in the env")
    except requests.RequestException as e:
        print(f"An error occurred during the health check: {e}")