import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from scipy.io import loadmat
from omics_model import MLP

DEVICE = torch.device("cpu")

st.set_page_config(page_title="Alzheimer’s Diagnosis System", layout="wide")
st.markdown("## 🧠 Alzheimer’s Diagnosis (EEG + Omics Ensemble)")

# ---------------------------
# Omics 관련 함수
# ---------------------------

@st.cache_resource
def load_omics_assets():
    try:
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        ckpt = torch.load("omics_mlp.pt", map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"❌ 파일 로드 실패: {e}")

    # 기본값 방어 (메타데이터 누락 시)
    in_dim = ckpt.get("in_dim", getattr(pca, "n_components_", 50))
    hidden = ckpt.get("hidden", 128)
    dropout = ckpt.get("dropout", 0.3)
    temperature = ckpt.get("temperature", 1.0)
    state_dict = ckpt.get("state_dict", ckpt)

    # 모델 생성 및 가중치 로드
    model = MLP(in_dim=int(in_dim), hidden=int(hidden), dropout=float(dropout))
    try:
        model.load_state_dict(state_dict)
    except Exception:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    return scaler, pca, model, float(temperature)


def predict_omics_from_soft(file) -> float:
    df = pd.read_csv(file, sep="\t", comment="!", index_col=0)
    scaler, pca, model, temp = load_omics_assets()

    raw = df.values.astype(float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = scaler.transform(raw)
    X_pca = pca.transform(X_scaled)

    X = torch.tensor(X_pca, dtype=torch.float32)
    logits = model(X) / temp
    probs = torch.sigmoid(logits).detach().numpy()

    return float(np.mean(probs))


# ---------------------------
# EEG 모델 정의
# ---------------------------

class EEG_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2))
        self.pool = nn.MaxPool2d((1, 2))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * 19 * 128, num_classes)

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

    if eeg.shape[1] == 19:
        eeg = eeg.T

    ch, t = eeg.shape
    epochs = []
    for start in range(0, t - window + 1, stride):
        epochs.append(eeg[:, start:start + window])
    return np.stack(epochs)


def predict_eeg_from_mat(file) -> float:
    model = load_eeg_model()
    arr = preprocess_eeg_mat(file)

    arr = (arr - arr.mean(axis=(1, 2), keepdims=True)) / (arr.std(axis=(1, 2), keepdims=True) + 1e-6)
    arr = np.nan_to_num(arr)

    X = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    logits = model(X)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return float(np.mean(probs[:, 0]))


# ---------------------------
# 앙상블 계산 + 출력
# ---------------------------

def risk_bucket(p):
    if p < 0.3:
        return "Low Risk", "green"
    elif p < 0.7:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"


def run_inference(omics_file, eeg_file):
    try:
        p_omics = predict_omics_from_soft(omics_file)
        p_eeg = predict_eeg_from_mat(eeg_file)
        p_final = 0.5 * p_omics + 0.5 * p_eeg
        risk, color = risk_bucket(p_final)
        return p_omics, p_eeg, p_final, risk, color
    except Exception as e:
        raise RuntimeError(f"❌ 추론 중 오류: {e}")


# ---------------------------
# Streamlit UI Layout
# ---------------------------

omics_file = st.file_uploader("🧬 Upload Omics Data (.soft / .tsv)", type=["soft", "tsv", "txt"])
eeg_file = st.file_uploader("💓 Upload EEG Data (.mat)", type=["mat"])

if st.button("🔍 Run Inference"):
    if not omics_file or not eeg_file:
        st.warning("⚠️ Omics와 EEG 파일을 모두 업로드하세요.")
    else:
        with st.spinner("모델 추론 중..."):
            try:
                p_omics, p_eeg, p_final, risk, color = run_inference(omics_file, eeg_file)
                st.success("✅ 추론 완료!")
                st.write(f"**EEG 확률(AD)**: {p_eeg:.4f}")
                st.write(f"**Omics 확률(AD)**: {p_omics:.4f}")
                st.write(f"**앙상블 최종 확률(AD)**: {p_final:.4f}")
                st.markdown(f"**위험도 등급:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))
