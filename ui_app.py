import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import tempfile

# ---------------------------
# Omics: 외부 코드 불러오기
# ---------------------------
from omics_model import predict_omics   # ← 그대로 불러와 사용

# ---------------------------
# EEG: CNN 정의 + 모델 로드
# ---------------------------
DEVICE = torch.device("cpu")

class EEG_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,5), padding=(1,2))
        self.conv2 = nn.Conv2d(16,32, kernel_size=(3,5), padding=(1,2))
        self.pool = nn.MaxPool2d((1,2))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32*19*128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

@st.cache_resource
def load_eeg_model():
    model = EEG_CNN().to(DEVICE)
    state = torch.load("best_model.pt", map_location=DEVICE)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

def preprocess_eeg_mat(file, window=512, stride=256):
    mat = loadmat(file)
    eeg = mat.get("export")
    if eeg is None:
        raise ValueError("❌ 'export' 키가 없습니다.")

    # (time, channel) → (channel, time)
    if eeg.shape[1] == 19:
        eeg = eeg.T   # (19, T)

    ch, t = eeg.shape
    epochs = []
    for start in range(0, t-window+1, stride):
        epochs.append(eeg[:, start:start+window])
    return np.stack(epochs)  # (N,19,512)

def predict_eeg_from_mat(file) -> float:
    model = load_eeg_model()
    arr = preprocess_eeg_mat(file)

    # Normalize
    arr = (arr - arr.mean(axis=(1,2), keepdims=True)) / (arr.std(axis=(1,2), keepdims=True) + 1e-6)
    arr = np.nan_to_num(arr)

    X = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    logits = model(X)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return float(np.mean(probs[:,0]))  # 클래스0=AD 확률 평균

# ---------------------------
# 앙상블 + UI
# ---------------------------
def risk_bucket(p):
    if p < 0.3: return "Low Risk","green"
    elif p < 0.7: return "Medium Risk","orange"
    return "High Risk","red"

st.title("🧠 Alzheimer’s Diagnosis System (EEG + Omics Ensemble)")
st.caption("EEG(.mat) + Omics(.soft) → 앙상블 (EEG=0.499, Omics=0.501)")

omics_file = st.file_uploader("Upload Omics Data (.soft)", type=["soft"])
eeg_file   = st.file_uploader("Upload EEG Data (.mat)", type=["mat"])

if st.button("🔍 Run Inference"):
    if not omics_file or not eeg_file:
        st.warning("두 파일을 모두 업로드하세요.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".soft") as tmp1:
            tmp1.write(omics_file.read()); omics_path = tmp1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp2:
            tmp2.write(eeg_file.read()); eeg_path = tmp2.name

        # 예측
        p_omics = predict_omics(omics_path)   # ← omics_model.py 사용
        p_eeg   = predict_eeg_from_mat(eeg_path)
        p_final = 0.499 * p_eeg + 0.501 * p_omics

        # 위험도
        risk,color = risk_bucket(p_final)

        # 결과 출력
        st.subheader("📊 결과")
        st.write(f"EEG 모델 확률(AD): **{p_eeg:.4f}**")
        st.write(f"Omics 모델 확률(AD): **{p_omics:.4f}**")
        st.write(f"앙상블 최종 확률(AD): **{p_final:.4f}**")
        st.progress(min(max(p_final,0.0),1.0))
        st.markdown(f"<h2 style='color:{color}'>{risk}</h2>", unsafe_allow_html=True)


