import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from scipy.io import loadmat

DEVICE = torch.device("cpu")

# ---------------------------
# Omics: 모델/전처리 로드
# ---------------------------
from omics_model import MLP

def load_omics_assets():
    scaler = joblib.load("scaler.pkl")
    pca    = joblib.load("pca.pkl")
    ckpt   = torch.load("omics_mlp.pt", map_location="cpu")

    model = MLP(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    temp = ckpt.get("temperature", 1.0)
    return scaler, pca, model, float(max(temp, 1e-3))

def predict_omics_from_soft(file) -> float:
    df = pd.read_csv(file.name, sep="\t", comment="!", index_col=0)
    scaler, pca, model, temp = load_omics_assets()

    raw = df.values.astype(float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = scaler.transform(raw)
    X_pca    = pca.transform(X_scaled)
    X        = torch.tensor(X_pca, dtype=torch.float32)

    logits = model(X) / temp
    probs  = torch.sigmoid(logits).numpy()
    return float(np.mean(probs))


# ---------------------------
# EEG: CNN 정의 + 모델 로드
# ---------------------------
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
    mat = loadmat(file.name)
    eeg = mat.get("export")
    if eeg is None:
        raise ValueError("❌ 'export' 키가 없습니다.")

    if eeg.shape[1] == 19:  # (T,19) → (19,T)
        eeg = eeg.T

    ch, t = eeg.shape
    epochs = []
    for start in range(0, t-window+1, stride):
        epochs.append(eeg[:, start:start+window])
    return np.stack(epochs)

def predict_eeg_from_mat(file) -> float:
    model = load_eeg_model()
    arr = preprocess_eeg_mat(file)

    arr = (arr - arr.mean(axis=(1,2), keepdims=True)) / (arr.std(axis=(1,2), keepdims=True) + 1e-6)
    arr = np.nan_to_num(arr)

    X = torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    logits = model(X)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return float(np.mean(probs[:,0]))  # 클래스0=AD 확률 평균


# ---------------------------
# 앙상블 + 위험도
# ---------------------------
def risk_bucket(p):
    if p < 0.3: return "Low Risk","green"
    elif p < 0.7: return "Medium Risk","orange"
    return "High Risk","red"

def run_inference(omics_file, eeg_file):
    if omics_file is None or eeg_file is None:
        return "파일을 모두 업로드하세요.", None, None, None, None

    p_omics = predict_omics_from_soft(omics_file)
    p_eeg   = predict_eeg_from_mat(eeg_file)
    p_final = 0.499 * p_eeg + 0.501 * p_omics

    risk,color = risk_bucket(p_final)

    return (
        f"EEG 확률(AD): {p_eeg:.4f}",
        f"Omics 확률(AD): {p_omics:.4f}",
        f"앙상블 최종 확률(AD): {p_final:.4f}",
        risk,
        color
    )


# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Alzheimer’s Diagnosis System (EEG + Omics Ensemble)")

    with gr.Row():
        omics_file = gr.File(label="Upload Omics Data (.soft)", type="file")
        eeg_file   = gr.File(label="Upload EEG Data (.mat)", type="file")

    run_btn = gr.Button("🔍 Run Inference")

    eeg_out   = gr.Textbox(label="EEG 모델 결과")
    omics_out = gr.Textbox(label="Omics 모델 결과")
    final_out = gr.Textbox(label="앙상블 최종 결과")
    risk_out  = gr.Textbox(label="위험도 등급")
    color_out = gr.Textbox(label="색상 코드")

    run_btn.click(
        fn=run_inference,
        inputs=[omics_file, eeg_file],
        outputs=[eeg_out, omics_out, final_out, risk_out, color_out]
    )

if __name__ == "__main__":
    demo.launch()
