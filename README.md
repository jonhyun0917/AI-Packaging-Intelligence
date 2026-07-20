# AI Packaging Intelligence Platform V9.1

AI 기반 포장 효율 분석, Void Space/PVI 계산, 추천 박스 및 PDF 보고서를 제공하는 Streamlit 애플리케이션입니다.

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. 이 폴더의 파일을 GitHub 저장소 루트에 업로드합니다.
2. Streamlit Community Cloud에서 `Create app`을 선택합니다.
3. GitHub 저장소와 브랜치를 선택합니다.
4. Main file path에 `app.py`를 입력합니다.
5. Deploy를 누릅니다.

## 배포 전 주의

- API 키, 개인정보, `.env`, `secrets.toml`은 GitHub에 올리지 마세요.
- 파일 경로는 프로젝트 루트 기준 상대경로를 사용하세요.
- 무료 배포 환경에서는 큰 모델과 대용량 파일이 메모리/저장공간 제한을 받을 수 있습니다.

## 프로젝트 구조

- `app.py`: Streamlit 진입점
- `analysis/`: PVI, ESG, 탄소 및 해석 로직
- `core/`: 검출, 측정, 품질 및 한계 평가
- `optimization/`: 박스·배치 추천
- `reporting/`: PDF 보고서와 인증서
- `database/`: 박스 데이터베이스
- `assets/`: 화면용 이미지 및 자산
