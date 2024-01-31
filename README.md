## 資料夾樹狀結構
```
Research_FINAL
├── CRIS_spectrum_sharing
│   ├── CreateData
│   ├── CRIS_BB    # Branch-and-bound方法驗證
│   │   └── channel_data
│   ├── CRIS_best_random    # Best Random方法驗證
│   │   └── channel_data
│   ├── CRIS_RL    # 本研究提出的RL方法
│   │   ├── channel_data
│   │   ├── checkpoints
│   │   │   └── models
│   │   └── gym_foo
│   │       └── envs
│   ├── CRIS_SA    # Heuristic中的Simulated annealing方法驗證
│   │   └── channel_data
│   ├── Generate_channel_data    # 生成通道CSI製造測試集以用來與其他方法做比較
│   │   └── gym_foo
│   │       └── envs
│   └── PLOT    # 結果呈現
│       ├── epsilon greedy
│       ├── hyper_parameter
│       │   ├── different_lr
│       │   ├── different_reg
│       │   └── different_weight
│       ├── model_compare
│       ├── reward_SE
│       ├── RIS_element
│       ├── spectrum_allocation
│       ├── stepwise_computing_time
│       └── stepwise_result
└── future_work    # 利用BB實做閒置頻段數量與用戶通道容量的關係比較
```

## 環境搭建
### Python
- virtualenv
    建立虛擬環境 `virtualenv <name> `
    進入 `source <name>/bin/activate`
    退出 `deactivate`
    
- packages (Use `pip install`)
    - [pytorch](https://pytorch.org/get-started/locally/)
        注意電腦的 CUDA 版本，並根據需要更改 shell 指令。
        - 如果 CUDA 版本為 11.6，執行以下指令：

        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
    - gym==0.21.0
    - pandas

### Branch-and-Bound
- [Couenne slover](https://www.coin-or.org/download/source/Couenne/)
    下載 `Couenne-0.5.8.zip`
    利用資料夾內的`README`, `INSTALL`按照步驟安裝
    ※ pyomo 套件建議用6.4.0版本

---

## 資料夾功能說明與使用方法
### 1. 生成訓練集與驗證集用戶資料
(a) 進入CreateData資料夾 - `cd CreateData` <br>
(b) 依據MDP生成主要用戶頻段使用情況 - `python Create_spectrum_MDP.py` <br>
(c\) 生成用戶位置資訊與主要用戶發射功率 - `python CreateUserTrajectory_MDP.py` <br>

執行完畢後就會得到包含頻譜資訊與用戶資訊的`.csv`檔

### 2. Generate_channel_data資料夾講解
(a) 進入Generate_channel_data資料夾 - `cd Generate_channel_data` <br>
(b) 執行程式 - `python generate_evaluation.py --RIS_N=XX` <br>
(c\) 生成測試資料集`.npy`檔 <br>


### 3. CRIS_RL資料夾講解

#### - CRIS_RL資料夾內部樹狀結構

```
CRIS_RL
├── channel_data     # 測試集放置的資料夾，使用`evaluation.py`會使用到
│   └── (Dataset from `Generate_channel_data/`)
├── DARC.py    # 連續動作輸出的DARC,DDPG,TD3模型
├── DDPG.py
├── TD3.py
├── globe.py
├── gym_foo    # 環境設定
│   ├── envs
│   │   ├── foo_env_improve.py    # 主要演算法設計
│   │   └── __init__.py
│   └── __init__.py
├── improved_DARC.py    # 混合動作輸出的HA-DARC,HA-DDPG,HA-TD3模型
├── improved_DDPG.py
├── improved_TD3.py
├── main.py    # 主要執行檔案
├── spectrum_sensing_MDP.py    # 頻譜感測
├── evaluation.py    # 測試集驗證結果
└── utils.py    # replay buffer的設置

```

#### - 執行程式
進入CRIS_RL資料夾 - `cd CRIS_RL/`
<shell指令可自行設定參數>
```python
python main.py --policy="DARC" --episdoe=30000 --actor-lr=5e-4 --critic-lr=5e-4 --RIS_N=12 --dir="result.txt"
```

#### - 測驗集測試
(a) `channel_data/`內放入由`Generate_channel_data/`生成的測試集 <br>
(b) 執行程式：指定RL演算法與反射元件數量
```python
python evaluation.py --policy="DARC" --RIS_N=12
```
就可得到測試集經由RL演算法的輸出結果`.csv`檔

### 4. CRIS_BB資料夾講解
(a) 進入CRIS_BB資料夾 - `cd CRIS_BB/` <br>
(b) `channel_data/`內放入由`Generate_channel_data/`生成的測試集 <br>
(c\) 執行程式 - `python BBSearch.py` <br>

就可得到測試集經由BB演算法的輸出結果`.csv`檔

### 5. CRIS_best_random資料夾講解
(a) 進入CRIS_best_random資料夾 - `cd CRIS_best_random/` <br>
(b) `channel_data/`內放入由`Generate_channel_data/`生成的測試集 <br>
(c\) 執行程式 - `python Best_random.py` <br>

就可得到測試集經由Best random的輸出結果`.csv`檔

### 6. CRIS_SA資料夾講解
(a) 進入CRIS_SA資料夾 - `cd CRIS_SA/` <br>
(b) `channel_data/`內放入由`Generate_channel_data/`生成的測試集 <br>
(c\) 執行程式 - `python simulated_annealing.py` <br>

就可得到測試集經由SA演算法的輸出結果`.csv`檔
