import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ---------------------------
# MLP 모델 정의
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(int(in_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), int(hidden))
        self.dropout = nn.Dropout(float(dropout))
        self.fc3 = nn.Linear(int(hidden), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


# ---------------------------
# Omics 학습 + 저장 함수
# ---------------------------
def train_and_save_model(data_path, save_model="omics_mlp.pt", save_scaler="scaler.pkl", save_pca="pca.pkl"):
    # 1️⃣ 데이터 불러오기 (.soft 파일 예시)
    df = pd.read_csv(data_path, sep="\t", comment="!", index_col=0)
    X = df.values.T  # (samples, genes)
    y = np.random.randint(0, 2, size=X.shape[0])  # 가짜 라벨 (임시)

    # 2️⃣ 전처리 (scaler + PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    # 3️⃣ 모델 정의
    in_dim = int(X_pca.shape[1])
    hidden = 128
    dropout = 0.3
