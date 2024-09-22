from transformers import BertTokenizer, BertModel
import torch

# 初始化BERT模型和分詞器
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 要處理的句子
sentences = ["今天天氣真好", "不想上班想請假玩", "地鐵又要罷工", "網絡很差總是斷線"]

# 將句子轉換為BERT的輸入格式
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 使用BERT模型
with torch.no_grad():
    outputs = model(**inputs)

# 獲取句子的特徵表示
sentence_vectors = outputs.last_hidden_state.mean(dim=1)
print(sentence_vectors)
