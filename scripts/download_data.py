import os
import requests
import zipfile
import shutil

def download_file(url, filepath):
    response = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def download_flickr30k(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    
    print("Manual download required:")
    print("1. Images: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset")
    print("2. Annotations: https://github.com/BryanPlummer/flickr30k_entities/archive/master.zip")
    print("3. Extract images to:", os.path.join(data_dir, "flickr30k_images"))
    print("4. Run script again after manual download")
    
    ann_zip = os.path.join(data_dir, "annotations.zip")
    
    if not os.path.exists(ann_zip):
        try:
            download_file("https://github.com/BryanPlummer/flickr30k_entities/archive/master.zip", ann_zip)
        except:
            print("Download annotations manually from GitHub")
    
    if not os.path.exists(os.path.join(data_dir, "flickr30k_images")):
        os.system(f"cd {data_dir} && tar -xf flickr30k-images.tar")
    
    if not os.path.exists(os.path.join(data_dir, "Annotations")):
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        src = os.path.join(data_dir, "flickr30k_entities-master")
        for folder in ["Annotations", "Sentences"]:
            if os.path.exists(os.path.join(src, folder)):
                shutil.move(os.path.join(src, folder), os.path.join(data_dir, folder))
    
    create_splits(data_dir)

def create_splits(data_dir):
    img_dir = os.path.join(data_dir, "flickr30k_images")
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    images.sort()
    
    total = len(images)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    splits = {
        'train.txt': images[:train_end],
        'val.txt': images[train_end:val_end],
        'test.txt': images[val_end:]
    }
    
    for split_file, img_list in splits.items():
        with open(os.path.join(data_dir, split_file), 'w') as f:
            for img in img_list:
                f.write(img + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()
    
    download_flickr30k(args.data_dir)