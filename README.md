# open-problems-multimodal

![header](https://github.com/riow1983/amex-default-prediction/blob/main/png/header.png)<br>
https://www.kaggle.com/competitions/amex-default-prediction<br>
どんなコンペ?:<br>
開催期間:<br>
![timeline](https://github.com/riow1983/amex-default-prediction/blob/main/png/timeline.png)<br>
[結果](#2022-08-24)<br>  
<br>
<br>
<br>
***
## 実験管理テーブル
https://wandb.ai/riow1983/amex-default-prediction?workspace=user-riow1983
|commitSHA|comment|Local CV|Public LB|
|----|----|----|----|
<br>

## Late Submissions
|commitSHA|comment|Local CV|Private LB|Public LB|
|----|----|----|----|----|
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
<br>


#### Documentation (incl. Tutorial)
|name|url|status|comment|
|----|----|----|----|
<br>

#### BBC (StackOverflow / StackExchange / Quora / Reddit / Others)
|name|url|status|comment|
|----|----|----|----|
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

#### 2022-08-17
コンペ用レポジトリ作成.<br>
<br>
<br>

#### 2022-08-24
結果はxxx/xxx (暫定) だった. <br>
![private lb image](https://github.com/riow1983/us-patent-phrase-to-phrase-matching/blob/main/png/result.png)
<br>
<br>
**どのように取り組み, 何を反省しているか**<br>
**My submissions について**<br>
**xxxについて**<br>
<br>
<br>
<br>
Back to [Top](#amex-default-prediction)



