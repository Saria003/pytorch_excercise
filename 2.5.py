import requests
from pathlib import Path
import zipfile
import pandas as pd

data_path = Path.cwd() # Current working directory.
image_path = data_path / "cats_dogs_images" 
    # The target folder where images will be extracted.
    # (Current directory + /cats_dogs_images)
url = "https://github.com/ArstmN/Pytorch_DL_FaraDars/raw/main/data/cats_dogs.zip"

# Check if exists:
if image_path.is_dir(): 
    print(f"{image_path} directory exists")
# Create if needed
else: 
    print(f"No {image_path} directory, making one...")
    image_path.mkdir(parents=True, exist_ok=True)
    # Download the zip file:
    with open(data_path / "cats_dogs.zip", "wb") as f:
        request = requests.get(url)
        print("Downloading image data")
        f.write(request.content)

    # Open the zip file:
    with zipfile.ZipFile(data_path / "cats_dogs.zip", "r") as zip_ref:
        print("Extracting cats and dogs images")
        zip_ref.extractall(image_path)
#----------------------------------------------------------------------
# Custom Dataset:
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from skimage import io

class MyDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        super(MyDataset, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = Path('.').joinpath(self.image_dir, self.annotations.iloc[index, 0])
            # row -> index, col -> 0  => To get rows and columns by position, not by label(That's loc)
            # ➡️ If index = 1 → 2nd row → filename dog2.jpg.
            # ➡️ self.annotations.iloc[1, 0] → 'dog2.jpg'.

            # iloc[index, 0] → gets the filename.
            # iloc[index, 1] → gets the label.

            ## Label based indexing(column name):
            # loc[index, 'filename']
            # loc[index, 'label']

        image = io.imread(image_path) 
        label = self.annotations.iloc[index, 1] # sotoon index shomare 1

        if self.transform:
            image = self.transform(image)

        return image, label

train_data = MyDataset(csv_file='C:/Users/Marabi/Desktop/Py/Faradars_Pytorch/cats_dogs_images/annotations.csv', image_dir='C:/Users/Marabi/Desktop/Py/Faradars_Pytorch/cats_dogs_images/cats_dogs')
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

train_features, train_label = next(iter(train_dataloader))
# print(f"{train_label}:\n {train_features}") # one sample

# Anotation file:
image, label = train_data.__getitem__(2)
print(f"label: {label}\nimage:\n{image}")