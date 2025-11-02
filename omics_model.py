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
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


# ---------------------------
# Omics 학습 + 저장 함수
# ---------------------------
def train_and_save_model(data_path, save_model="omics_mlp.pt", save_scaler="scaler.pkl", save_pca="pca.pkl"):
    # 1. 데이터 불러오기 (.soft 파일 예시)
    df = pd.read_csv(data_path, sep="\t", comment="!", index_col=0)

    # 라벨이 별도로 필요하다면 불러오기 (예: AD=1, Control=0)
    # 여기서는 예시로 임의로 만든다고 가정
    y = np.random.randint(0, 2, size=df.shape[1])  # 샘플 라벨
    X = df.values.T  # (samples, genes)

    # 2. 전처리
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=50)  # PCA 차원 수
    X_pca = pca.fit_transform(X_scaled)

    # Tensor 변환
    X_tensor = torch.tensor(X_pca, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # 3. 모델 정의
    in_dim = X_pca.shape[1]   # PCA 출력 차원 (50)
    hidden = 128
    dropout = 0.3
    model = MLP(in_dim=in_dim, hidden=hidden, dropout=dropout)

    # 4. 학습 설정
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # 5. 학습 루프 (간단히 5 epochs만)
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    # 6. 모델 저장 (메타데이터 포함)
    checkpoint = {
        "in_dim": in_dim,
        "hidden": hidden,
        "dropout": dropout,
        "temperature": 1.0,
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, save_model)
    print(f"✅ 모델 저장 완료: {save_model}")

    # 7. Scaler & PCA 저장
    joblib.dump(scaler, save_scaler)
    joblib.dump(pca, save_pca)
    print(f"✅ Scaler 저장: {save_scaler}, PCA 저장: {save_pca}")


if __name__ == "__main__":
    # 예시 실행
    train_and_save_model("GSE33770_family.soft.gz")
