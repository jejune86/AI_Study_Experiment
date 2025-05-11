import whisper

# 1. 모델 로드 (캐시 경로 지정)
model = whisper.load_model("large-v3", download_root=".cache/whisper")

# 2. 오디오 인식 (언어 강제 설정)
result = model.transcribe(
    "sample.wav",
    language="ko",
    temperature=0.0,
    fp16=False,
    condition_on_previous_text=False,
    suppress_tokens=[]  # 아무것도 억제하지 않음
)

# 3. 출력
print("\n📄 STT 결과:")
print(result["text"])
