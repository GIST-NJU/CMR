from urllib.request import urlretrieve
from tqdm import tqdm
import os
import zipfile
import hashlib
import shutil


def download_url(url, destination=None, progress_bar=True):
    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        urlretrieve(url, filename=destination)


def extract_file(zip_path, target_files, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if isinstance(target_files, str) and target_files == 'all':
            target_files = zip_ref.namelist()
        elif isinstance(target_files, str):
            target_files = [target_files]
        for file in tqdm(target_files, desc=f'Extracting from {os.path.basename(zip_path)}'):
            zip_ref.extract(file, path=extract_to)


def md5sum(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        total = os.path.getsize(path) // chunk_size + 1
        for chunk in tqdm(iter(lambda: f.read(chunk_size), b""), desc=f'Checking MD5 for {os.path.basename(path)}', total=total):
            md5.update(chunk)
    return md5.hexdigest()


def download_COCO_files(target_dir='data/source/COCO'):
    coco_urls = {
        'train2014.zip':                'http://images.cocodataset.org/zips/train2014.zip',
        'val2014.zip':                  'http://images.cocodataset.org/zips/val2014.zip',
        'test2014.zip':                 'http://images.cocodataset.org/zips/test2014.zip',
        'image_info_test2014.zip':      'http://images.cocodataset.org/annotations/image_info_test2014.zip',
        'annotations_trainval2014.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    }
    coco_md5 = {
        'train2014.zip':                '0da8c0bd3d6becc4dcb32757491aca88',
        'val2014.zip':                  'a3d79f5ed8d289b7a7554ce06a5782b3',
        'test2014.zip':                 '04127eef689ceac55e3a572c2c92f264',
        'image_info_test2014.zip':      '25304cbbafb2117fb801c0f7218fdbba',
        'annotations_trainval2014.zip': '0a379cfc70b0e71301e0f377548639bd',
    }
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename, url in coco_urls.items():
        destination = os.path.join('tmp', f'{filename}')
        if not os.path.exists(destination) or md5sum(destination) != coco_md5[filename]:
            download_url(url, destination=destination)
            if md5sum(destination) != coco_md5[filename]:
                raise Warning(f'MD5 checksum mismatch for {filename}')
    for file in coco_urls.keys():
        zip_path = os.path.join('tmp', f'{file}')
        if file == 'annotations_trainval2014.zip':
            target_files = [
                'annotations/instances_train2014.json',
                'annotations/instances_val2014.json'
            ]
        else:
            target_files = 'all'
        extract_file(zip_path, target_files=target_files, extract_to=target_dir)
        if file == 'image_info_test2014.zip':
            shutil.copy(os.path.join(target_dir, 'annotations', 'image_info_test2014.json'), os.path.join(target_dir, 'image_info_test2014.json'))
            shutil.rmtree(os.path.join(target_dir, 'annotations'))
        elif file == 'annotations_trainval2014.zip':
            shutil.copy(os.path.join(target_dir, 'annotations', 'instances_train2014.json'), os.path.join(target_dir, 'instances_train2014.json'))
            shutil.copy(os.path.join(target_dir, 'annotations', 'instances_val2014.json'), os.path.join(target_dir, 'instances_val2014.json'))
            shutil.rmtree(os.path.join(target_dir, 'annotations'))


def download_zenodo_files(target_dir='./'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    zenodo_urls = {
        'data.zip':    'https://zenodo.org/record/18303101/files/data.zip',
        'models.zip':  'https://zenodo.org/record/18303101/files/models.zip',
        'results.zip': 'https://zenodo.org/record/18303101/files/results.zip'
    }
    zenodo_md5 = {
        'data.zip':    'd0041641c4f0cd1d93f166f5668561ac',
        'models.zip':  '9892ca7f1188bdcd5a5d503906c924a4',
        'results.zip': 'f476b2ddfe7f3a1547a239de424aa0d2'
    }
    for filename, url in zenodo_urls.items():
        destination = os.path.join('tmp', filename)
        if not os.path.exists(destination) or md5sum(destination) != zenodo_md5[filename]:
            download_url(url, destination=destination)
            if md5sum(destination) != zenodo_md5[filename]:
                raise Warning(f'MD5 checksum mismatch for {filename}')
    for file in zenodo_urls.keys():
        zip_path = os.path.join('tmp', file)
        extract_file(zip_path, target_files='all', extract_to=target_dir)


if __name__ == "__main__":
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    download_zenodo_files()
    download_COCO_files()
