"""
A simple-comprehensive method for dataset pulling.     
"""
import requests
import shutil

def pull_data_from_url(url: str, output_dir: str):
    with requests.get(url, stream=True) as r:
        with open(output_dir, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

if __name__ == '__main__':
    url = '' # URL for downloading the data
    output_dir = 'assets/raw_data.csv'
    
    pull_data_from_url(url = url, output_dir=output_dir)
