import { CodeSection, StepType } from './types';

export const CODE_SNIPPETS: Record<StepType, CodeSection> = {
  [StepType.OVERVIEW]: {
    id: 'overview',
    title: '完整流程概览',
    description: '整个项目分为三个核心阶段：大模型推理、批量处理、以及基于向量的评分评估。',
    code: `# 核心流程伪代码
def main():
    # 1. 准备数据
    train_data = load_csv('./train_data.csv')
    
    # 2. 大模型批量推理
    predictions = []
    for symptom in train_data:
        result = call_large_model(symptom)
        predictions.append(result)
        
    # 3. 评估结果 (向量相似度)
    final_score = score(predictions, ground_truth)
    print(f"最终得分: {final_score}")`
  },
  [StepType.LLM_INFERENCE]: {
    id: 'llm',
    title: '大模型推理 (LLM Inference)',
    description: '核心函数 call_large_model 负责构造 Prompt 并调用 ERNIE 模型，获取 JSON 格式的诊断结果。',
    code: `def call_large_model(symptoms):
    """
    调用ERNIE X1大模型进行证型治法生成
    """
    system_prompt = """
    你是一位经验丰富的中医专家，请根据患者症状描述判断：
    1. 证型：需符合慢性淋巴细胞白血病中医辨证标准
    2. 治法：需与证型对应且符合中医治疗原则
    请用JSON格式输出：{"证型":"", "治法":""}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"患者症状：{symptoms}"}
    ]

    try:
        response = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            messages=messages,
            # ... parameters
        )
        content = response.choices[0].message.content
        return parse_response(content)
    except Exception as e:
        return {"证型": "未知", "治法": "待定"}`
  },
  [StepType.BATCH_PROCESSING]: {
    id: 'batch',
    title: '批量处理循环',
    description: '遍历数据集，对每一条症状数据进行预测，并收集结果用于后续评估。',
    code: `# 读取数据
with open(train_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    # ... 读取逻辑 ...

# 预测循环
predict_zx = []
predict_zf = []

for symptom in symptoms_data:
    # 调用模型
    zx, zf = predict(symptom)
    
    # 存储结果
    predict_zx.append(zx)
    predict_zf.append(zf)
    
    # 避免API速率限制
    time.sleep(0.2)`
  },
  [StepType.EMBEDDING_METRICS]: {
    id: 'embedding',
    title: '向量化与相似度评分',
    description: '由于生成的文本可能不完全一致，我们使用 Embedding 将文本转化为向量，计算余弦相似度来评估准确性。',
    code: `def get_embedding(text):
    # 调用百度 Embedding API
    # 返回 768/1024 维度的向量列表
    # ... request code ...
    return data['data'][0]['embedding']

def calculate_similarity(text1, text2):
    # 计算两个文本向量的余弦相似度
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    sim = cosine_similarity([emb1], [emb2])
    return float(sim[0][0])

def score(predict_zx, predict_zf):
    # 计算证型和治法的平均相似度
    # final_score = ((zx_mean + zf_mean) / 2) * 100
    return final_score`
  }
};

export const MOCK_SYMPTOMS = [
    "患者头晕目眩，耳鸣，面赤，口苦，烦躁易怒，失眠多梦，舌红苔黄，脉弦数。",
    "面色苍白，神疲乏力，气短懒言，食少便溏，舌淡苔白，脉细弱。",
    "腰膝酸软，头晕耳鸣，五心烦热，骨蒸潮热，盗汗，舌红少苔，脉细数。"
];
