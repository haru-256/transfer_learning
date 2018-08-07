
# coding: utf-8

# # Food 101 のデータを学習データとテストデータに分ける

# In[46]:


import shutil
import pathlib
import pandas
from sklearn import model_selection


# In[47]:


data_dir = pathlib.Path("../data/food-101/images/").resolve()
data_dir


# In[48]:


train_dir = data_dir / "train"
val_dir= data_dir / "val"
test_dir = data_dir / "test"

for path in [train_dir, val_dir, test_dir]:
    if not path.exists():
        path.mkdir()
        print("make directory", path)


# In[49]:


data_dir = pathlib.Path("../data/food-101/images/datas/").resolve()
train_data_path_file = pathlib.Path("../data/food-101/meta/train.txt").resolve()
test_data_path_file = pathlib.Path("../data/food-101/meta/test.txt").resolve()


# ## 学習データの移動

# In[50]:


with open(train_data_path_file, "r") as file:
    for path in file:
        # pathの読み込み
        path = data_dir / "{}.jpg".format(path[:-1])  # 画像へのパス
        class_dir = train_dir / path.parts[-2]  # クラスラベルのディレクトリ
        if not class_dir.exists():
            class_dir.mkdir()
            
        # 画像ファイルの移動
        shutil.move(path, train_dir / "/".join(path.parts[-2:]))


# ## テストデータの移動

# In[51]:


with open(test_data_path_file, "r") as file:
    for path in file:
        # pathの読み込み
        path = data_dir / "{}.jpg".format(path[:-1])  # 画像へのパス
        class_dir = test_dir / path.parts[-2]  # クラスラベルのディレクトリ
        if not class_dir.exists():
            class_dir.mkdir()
        # 画像ファイルの移動
        shutil.move(path, test_dir / "/".join(path.parts[-2:]))

