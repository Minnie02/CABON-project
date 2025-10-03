#정환이형이 보내준거에서 수정된 상태
# omics_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)   # 이진 분류 (AD vs Control)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.out(x).squeeze(-1)  # 로짓 출력


def predict_omics(raw_data: np.ndarray, model, scaler, pca, T=1.0):
    """
    raw_data : (n_samples, n_features) numpy array (비전처리 원본)
    return   : 확률 (AD 위험도)
    """
    # 1) NaN/Inf 처리
    raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Scaling
    X_scaled = scaler.transform(raw_data)

    # 3) PCA
    X_proc = pca.transform(X_scaled)

    # 4) Tensor 변환
    X_tensor = torch.tensor(X_proc, dtype=torch.float32)

    # 5) 모델 추론 + 온도 보정
    with torch.no_grad():
        logits = model(X_tensor)
        logits = logits / max(T, 1e-3)   # Temperature scaling
        probs = torch.sigmoid(logits).numpy()

    return probs


# ---------------------------------------------------
# ✅ 수정: 독립 실행 모드에서만 동작하도록 분리
# ---------------------------------------------------
if __name__ == "__main__":   # ✅ 추가
    # 전처리기 불러오기
    scaler = joblib.load("scaler.pkl")
    pca    = joblib.load("pca.pkl")

    # 모델 불러오기
    checkpoint = torch.load("omics_mlp.pt", map_location="cpu")
    in_dim  = checkpoint["in_dim"]
    hidden  = checkpoint["hidden"]
    dropout = checkpoint["dropout"]
    T       = checkpoint.get("temperature", 1.0)

    model = MLP(in_dim=in_dim, hidden=hidden, dropout=dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("✅ omics_model standalone test OK")

    # ✅ 테스트용 코드 (임시 더미 데이터)
    dummy_data = np.random.rand(1, in_dim)   # 1개 샘플
    result = predict_omics(dummy_data, model, scaler, pca, T)
    print("예측 확률:", result)

     