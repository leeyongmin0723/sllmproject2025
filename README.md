# sllmproject2025
**캡스톤프로젝트2 - 프로젝트3: sLLM 만들기**

---

## 📚 프로젝트 소개

> "sLLM 만들기"는 한국어를 포함한 다양한 자연어 처리(NLP) 작업을 수행할 수 있는  
> **경량화된 대형 언어 모델(small Large Language Model, sLLM)** 구축 프로젝트입니다.

- **일상 대화**, **문서 생성**, **질문응답** 등 다양한 작업에 적합한 sLLM 개발
- 프롬프트(지시문) 기반 인스트럭션 튜닝 적용
- 한국어 및 다국어 지원
- Hugging Face 🤗 Transformers 라이브러리 활용

---

## 🧠 사용 모델

### [`openchat/openchat_3.5`](https://huggingface.co/openchat/openchat_3.5)

  "너는 유용한 AI야" 같은 지시문(Instruction)을 잘 이해하고 반응합니다.
- **대화형 최적화**  
  일상 대화, 문서 작성, Q&A, 아이디어 브레인스토밍 등에 적합합니다.
- **오픈소스 & 손쉬운 활용**  
  Hugging Face 🤗 Transformers에서 바로 불러와 사용 가능하며,  
  PyTorch 기반으로 파인튜닝과 커스터마이징이 용이합니다.

---
## 🔧 기술 스택

| 항목        | 내용                                 |
|-------------|--------------------------------------|
| 언어 모델    | openchat/openchat_3.5                |
| 백엔드       | FastAPI                             |
| 프론트엔드   | HTML + JavaScript (Vanilla)         |
| 실행 환경    | PyTorch + Hugging Face + **GPU (CUDA)**  ✅

- 본 프로젝트는 PyTorch 기반의 Hugging Face 모델을 GPU 환경에서 실행하며,
CPU 환경에서는 실행 속도가 현저히 느려질 수 있습니다.

---
## ✨ 특징

| 항목 | 설명 |
| :--- | :--- |
| 언어 지원 | 한국어 및 다양한 언어 |
| 튜닝 방식 | Instruction Tuning |
| 활용 분야 | 일상 대화, 문서 작성, 질문응답, 요약, 브레인스토밍 등 |
| 프레임워크 | PyTorch, Hugging Face Transformers |
| 모델 최적화 | 빠른 응답, 부드러운 대화 흐름 |



---

## 🎮 실행 화면 예시

### 📌 한국어로 질문했을 때

> **질문내용**: `삼성전자에 대해 알려줘`
![korean 3b](https://github.com/user-attachments/assets/09f0456d-badf-4c1b-9b47-657743ed0fee)
>
> ---


### 📌 영어로 질문했을 때

> **질문내용**: `Tell me about Samsung Electronics`
![english 3b](https://github.com/user-attachments/assets/f6e25687-6a69-4fcd-8942-8e5291fa5ab9)
