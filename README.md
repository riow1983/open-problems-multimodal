# open-problems-multimodal

![header](https://github.com/riow1983/open-problems-multimodal/blob/main/png/header.png)<br>
https://www.kaggle.com/competitions/open-problems-multimodal<br>
どんなコンペ?:<br>
開催期間:<br>
![timeline](https://github.com/riow1983/open-problems-multimodal/blob/main/png/timeline.png)<br>
[Saturn Cloud](https://app.community.saturnenterprise.io/dash/o/community/resources/jupyterServer/91adbd53412f4b1ab6375b986f71e2c5/)<br>
[結果](#2022-11-15)<br>  
<br>
<br>
<br>
***

## 実験管理テーブル
https://wandb.ai/riow1983/open-problems-multimodal?workspace=user-riow1983
|commitSHA|comment|W&B|Local CV|Public LB|Private LB|
|----|----|----|----|----|----|
|e335712825487cde604ce1d8c8490c6b8aa7575b|[Submit] cite-exp001|[URL](https://wandb.ai/riow1983/open-problems-multimodal/runs/wd1t294h/overview?workspace=user-riow1983)|0.7453(cite);?(multi)|0.793|?|
<br>

## Late Submissions
|commitSHA|comment|W&B|Local CV|Public LB|Private LB|
|----|----|----|----|----|----|
<br>


## My Assets
[notebook命名規則]  
- kagglenb001{e,t,i}-hoge.ipynb: Kaggle platform上で新規作成されたKaggle notebook (kernel).
- nb001{e,t,i}-hoge.ipynb: localで新規作成されたnotebook. 
- {e:EDA, t:train, i:inference}
- kaggle platform上で新規作成され, localで編集を加えるnotebookはファイル名kagglenbをnbに変更し, 番号は変更しない.

#### Code
作成したnotebook等の説明  
|name|url|status|comment|
|----|----|----|----|
<br>





***
## 参考資料
#### Snipets
```python
comp_name = "amex-default-prediction"
nb_name = 'kagglenb003-adversarial-validation'

import sys
import os
from pathlib import Path

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    base = f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/notebooks"
    %cd {base}

KAGGLE_ENV = True if 'KAGGLE_URL_BASE' in set(os.environ.keys()) else False
INPUT_DIR = Path('../input/')

if KAGGLE_ENV:
    OUTPUT_DIR = Path('')
else:
    !mkdir ../input/{nb_name}
    OUTPUT_DIR = INPUT_DIR / nb_name
```
<br>

```Python
class CFG(object):
    def __init__(self):
        self.debug = False
        self.params = {
            'loss_function' : 'Logloss',
            'eval_metric' : 'AUC',
            'learning_rate': 0.08,
            'num_boost_round': 5000,
            'early_stopping_rounds': 100,
            'random_state': 127,
            'task_type': 'GPU'
        }
        self.target = 'private'
    
        self.drop_cols = ['S_2', 'month', 'customer_ID', 'fold', self.target]

        self.num_rows = None
        if self.debug:
            self.num_rows = 1000


args = CFG()
print(args.num_rows)
```
<br>

```python
# Install cudf on Colab
# Credits to: 
# https://colab.research.google.com/drive/1xnTpVS194BJ0pOPuxN4GOmypdu2RvwdH


# Cell #0
# This get the RAPIDS-Colab install files and test check your GPU.  Run this and the next cell only.
# Please read the output of this cell.  If your Colab Instance is not RAPIDS compatible, it will warn you and give you remediation steps.
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/env-check.py

# Cell #1
# This will update the Colab environment and restart the kernel.  Don't run the next cell until you see the session crash.
!bash rapidsai-csp-utils/colab/update_gcc.sh
import os
os._exit(00)

# Cell #2
# This will install CondaColab.  This will restart your kernel one last time.  Run this cell by itself and only run the next cell once you see the session crash.
import condacolab
condacolab.install()

# Cell #3
# you can now run the rest of the cells as normal
import condacolab
condacolab.check()

# Cell #4 (This will take about 15 minutes.)
# Installing RAPIDS is now 'python rapidsai-csp-utils/colab/install_rapids.py <release> <packages>'
# The <release> options are 'stable' and 'nightly'.  Leaving it blank or adding any other words will default to stable.
!python rapidsai-csp-utils/colab/install_rapids.py stable
import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
os.environ['CONDA_PREFIX'] = '/usr/local'


# Cell #5
import cudf
import cupy

import cffi
print(cffi.__version__)

!pip uninstall cffi
!pip install cffi==1.15.0

import importlib
importlib.reload(cffi)

print(cffi.__version__)
```
<br>

```bash
# Google Drive path
cd /Volumes/GoogleDrive/My\ Drive/colab_notebooks/kaggle/open-problems-multimodal/
```
<br>


#### Papers
|name|url|status|comment|
|----|----|----|----|
<br>


#### Blogs (Medium / Qiita / Others)
|name|url|status|comment|
|----|----|----|----|
|Python の super() 関数の使い方|[URL](https://www.lifewithpython.com/2014/01/python-super-function.html)|Read|super()の引数にクラス名を渡すのはPython2の記法.|
|【Mac】SSHログイン中に「client_loop: send disconnect: Broken pipe」でフリーズ・ログアウトする事象の解決方法|[URL](https://genchan.net/it/pc/mac/11402/)|Read|効果無し.|
|YAML Tutorial Quick Start: A Simple File|[URL](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started)|Read|YAMLファイルの記法;参考になる.|
<br>


#### Documentation (incl. Tutorial)
|name|url|status|comment|
|----|----|----|----|
<br>

#### BBC (StackOverflow / StackExchange / Quora / Reddit / Others)
|name|url|status|comment|
|----|----|----|----|
|How can I parse a YAML file in Python|[URL](https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python)|Read|YAMLファイルの読み方;参考になる.|
<br>

#### GitHub
|name|url|status|comment|
|----|----|----|----|
<br>

#### Hugging Face
|name|url|status|comment|
|----|----|----|----|
<br>

#### Colab Notebook
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Notebooks)
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Datasets)
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle (Discussion)
|name|url|status|comment|
|----|----|----|----|
<br>



***
## Diary

#### 2022-9-30
According to [this notebook](https://www.kaggle.com/code/takanashihumbert/there-are-35-features-in-multiome-data-constant-0/notebook), constant columns in multiome train are:
```python
const_cols_train = set(['chr10:41857982-41858833', 'chr10:41879051-41879911', 'chr11:1946784-1947675', 'chr13:65965314-65966055', 'chr21:21146641-21147535', 'chr2:174378385-174378803', 'chr3:56842415-56843062', 'chr4:131727166-131728039', 'chr9:117692728-117693552', 'chr9:35269996-35270417'])
```
while constant columns in multiome test are:
```python
const_cols_test = set(['chr10:79753665-79753951', 'chr11:115700218-115700585', 'chr11:29329177-29329459', 'chr11:42407098-42407984', 'chr12:126509499-126510333', 'chr14:32622946-32623255', 'chr16:87733202-87733436', 'chr1:4058887-4059409', 'chr20:59565631-59566004', 'chr2:102597677-102597978', 'chr2:89307028-89307866', 'chr3:11677056-11677358', 'chr3:55631693-55632564', 'chr3:8336689-8336899', 'chr3:98231236-98231837', 'chr4:122370131-122370487', 'chr4:123042897-123043148', 'chr4:144557386-144557968', 'chr5:13477356-13477709', 'chr5:87204838-87205174', 'chr5:877133-877370', 'chr6:148829090-148829417', 'chr6:167115297-167115502', 'chr7:102606719-102606960', 'chr7:52501818-52502338'])
```
And the intersection of the two has none:
```python
print(const_cols_train.intersection(const_cols_test))
# set()
```
This means there is no column that can be dropped because of its consistency across train and test in multiome, which is opposite to the case of cite where many columns have been dropped because of their consistency across train and test.
<br>
<br>
<br>

#### 2022-11-15
結果はxxx/xxx (暫定) だった. <br>
![private lb image](https://github.com/riow1983/open-problems-multimodal/blob/main/png/result.png)
<br>
<br>
**どのように取り組み, 何を反省しているか**<br>
**My submissions について**<br>
**xxxについて**<br>
<br>
<br>
<br>
Back to [Top](#open-problems-multimodal)



