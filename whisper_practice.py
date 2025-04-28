import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# 모델, 프로세서 로드
model_name = "openai/whisper-large-v2"  
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.eval()

# 오디오 파일 불러오기
audio, sr = torchaudio.load("sample.wav")
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
audio = audio.squeeze()

# 입력 변환
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")

# 디코딩 
with torch.no_grad():
    generated = model.generate(
        input_features, 
        forced_decoder_ids=forced_decoder_ids,
        output_scores=True,
        return_dict_in_generate=True
    )


predicted_ids = generated.sequences[0]
tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids)

transcription = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
print(f"\nSTT 결과: {transcription}\n")


scores = generated.scores  

# token별 confidence 계산
confidences = []
for step_logits, token_id in zip(scores, predicted_ids[1:]):  # BOS는 제외
    log_probs = step_logits.log_softmax(dim=-1)
    token_log_prob = log_probs[0, token_id]
    confidence = token_log_prob.exp().item()
    confidences.append(confidence)

# 결과 출력
decoded_tokens = []

for token_id in predicted_ids[1:]:  # BOS 제외
    decoded_token = processor.tokenizer.decode([token_id], skip_special_tokens=True)
    decoded_tokens.append(decoded_token)

for token, conf in zip(decoded_tokens, confidences):
    print(f"Token: {token}, Confidence: {conf:.4f}")


import numpy as np

def compute_pronunciation_score(confidences, outlier_std_threshold=1.0):

    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)

    # outlier 기준 설정
    threshold = mean_conf - outlier_std_threshold * std_conf

    # outlier를 제거한 confidence 필터링
    filtered_confidences = [c for c in confidences if c >= threshold]

    if filtered_confidences:
        final_score = np.mean(filtered_confidences)
    else:
        final_score = 0.0  # 모두 제거됐으면 0으로 처리

    return final_score, len(filtered_confidences), threshold

final_score, num_used, used_threshold = compute_pronunciation_score(confidences)
print(f"최종 발음 점수: {final_score:.4f} (사용한 토큰 수: {num_used}, 기준 threshold: {used_threshold:.4f})")
