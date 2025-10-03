# Alzheimer EEG — Streamlit App

**EEG(.mat) 업로드 → 서버 전처리 → PyTorch 모델 추론 → 확률 시각화**를 한 파일로 구성한 Streamlit UI입니다.

## 빠른 시작 (로컬)
```bash
pip install -r requirements.txt
streamlit run app.py
```
- 기본적으로 `artifacts/`에 있는 `model.pt`(TorchScript) 또는 `model_state.pth`(state_dict)를 로드합니다.
- 학습 시의 전처리 파라미터/클래스 순서를 `app.py` 상단 설정에서 반드시 동일하게 맞추세요.

## 배포 (Streamlit Community Cloud)
1. 이 프로젝트를 GitHub에 푸시
2. https://share.streamlit.io 에서 새 앱 생성 (repo/branch/app.py 선택)
3. `artifacts/` 안의 모델 파일이 리포지토리에 포함되어 있어야 합니다. (용량 제한 참고)
4. 필요 시 `secrets`에 환경변수/URL 등을 등록할 수 있습니다.

## 주의
- 연구/교육용 데모이며, 의료적 판단에 사용하면 안 됩니다.
- 업로드 데이터에 개인식별정보가 포함되지 않도록 유의하세요.
