import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# GPU 사용 설정 => 여기는 지워도 상관 없을 듯
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 여기가 Input 넣는 곳
with open('Naver_Review1000.json', 'r', encoding='utf-8') as file:
    reviews = json.load(file)

# 로컬에 저장된 모델과 토크나이저 경로
model_path = 'kobert_sentiment_model_last'  # 또는 'kobert_sentiment_model_last'

# 모델과 토크나이저 로드
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(model_path)

# 새로운 리뷰 텍스트에 대한 점수 예측 함수
def predict_score(review_text):
    encoding = tokenizer(review_text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    return prediction

# 예시 리뷰
# review = "이 식당의 음식은 정말 맛있어요!"


# Output 넣는 곳
results = []
for review in reviews:
    review_text = review['body']
    prediction = predict_score(review_text)
    results.append({
        'review': review_text,
        'predicted_score': prediction
    })

with open('Result.json', 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, indent=4, ensure_ascii=False)