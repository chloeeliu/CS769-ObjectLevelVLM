import os
import yaml
import urllib.request
import tarfile

# Load configuration
with open("configs/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

DATA_FOLDER = config['DATA_FOLDER']
train_image_path = os.path.join(DATA_FOLDER, "CXR8")
os.makedirs(train_image_path, exist_ok=True)

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]


def download_and_extract(url, dest_folder):
    fn = url.split('/')[-1]
    location = os.path.join(dest_folder, fn)
    print(f'Downloading {fn}...')
    urllib.request.urlretrieve(url, location)

    try:
        with tarfile.open(location, "r:gz") as tar:
            tar.extractall(path=dest_folder)
    finally:
        os.remove(location)

from concurrent.futures import ThreadPoolExecutor

# Using ThreadPoolExecutor to download and extract files concurrently
with ThreadPoolExecutor(max_workers=5) as executor:
    for link in links:
        executor.submit(download_and_extract, link, train_image_path)


