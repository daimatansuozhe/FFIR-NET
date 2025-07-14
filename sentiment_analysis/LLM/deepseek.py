import pandas as pd
import openai
from tqdm import tqdm
from sklearn.metrics import classification_report


def sentiment_analysis(text, time,name,sys_prompt="You are a helpful assistant",custom_prompt=None, model="deepseek-chat", temperature=0.3):
    #默认用户提示词
    if custom_prompt is None:
        custom_prompt = (
            "请结合句子发布的时间和股票名称，判断以下句子的情感倾向（积极则label为2、消极则label为0、中性则label为1），并给出情感得分：\n"
            f"句子：{text}\n"
            f"发布时间{time}\n"
            f"股票名称：{name}\n"
            "label:"
            "score："
            "格式要求如下：text：xxx\nlabel：xxx\nscore：xxx"
            "严格按照格式输出，禁止输出无关内容"
        )
    else:
        custom_prompt = custom_prompt.format(text=text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": custom_prompt}
        ],
        temperature=temperature
    )
    #返回str：“123”
    return response.choices[0].message.content.strip()

# 设置 client:调用的api_key和api——url,custom:用户提示词,sys:系统提示词
client = openai.OpenAI(api_key="sk-11da4c508da341088bbf4ec4afc66bf8",base_url='https://api.deepseek.com')


input_path= '../dataset/other/2023_1_2025_1/post_info.post_000333.csv'


df=pd.read_csv(input_path)
# custom = (
#     "请以简洁的方式分析这段话的情绪倾向（positive、negative、neutral）以及情感得分（置信度）保留三位小数：\n{text}\n"
#     "请输出格式如下：text：xxx\nlabel：xxx\nscore：xxx"
# )
sys="你现在是金融领域的情感分析专家"
temperature=0.3

#情感分析
result = []
name = "美的集团"
batch_size = 1000
batch_num = 0
def parse_result(result_str):
    """解析API返回的情感分析结果字符串为字典格式"""
    entry = {}
    lines = result_str.split('\n')
    for line in lines:
        if '：' in line:  # 中文冒号分隔
            key, value = line.split('：', 1)  # 只分割第一个冒号
            entry[key.strip()] = value.strip()
        elif ':' in line:  # 英文冒号分隔
            key, value = line.split(':', 1)
            entry[key.strip()] = value.strip()
    return entry
for i, (texts, time) in tqdm(enumerate(zip(df['post_title'], df['post_time'])), total=len(df)):
    result.append(sentiment_analysis(texts, name=name, time=time, temperature=temperature))

    # 每1000条保存一次
    if (i + 1) % batch_size == 0:
        batch_num += 1
        res = pd.DataFrame([parse_result(r) for r in result])
        res.to_csv(f"./out/pre_batch_{batch_num}.csv", index=False)
        result = []  # 清空当前批次

# 保存剩余不足1000条的数据
if result:
    res = pd.DataFrame([parse_result(r) for r in result])
    res.to_csv(f"./out/pre_batch_{batch_num + 1}.csv", index=False)
# #结果处理成dict
# result2dic=[]
# for i in result:
#     item={}
#     lines=i.split("\n")
#     for line in lines:
#         if line=="":
#             continue
#         else:
#             key, value = line.split('：', 1)
#             item[key.strip()]=value.strip()
#     result2dic.append(item)
#
# #先转df再转csv
# res=pd.DataFrame(result2dic)
# res.to_csv("./out/pre.csv",index=False)



# #准确率
# acc=0
# correct=0
# # 情感映射表
# label_map = {
#     "negative": 0,
#     "neutral" : 1,
#     "positive": 2
#
# }
#
# # 新增数值列
# res["label"] = res["label"].map(label_map)
# y_pre = res["label"].tolist()
# y_label = df["label"].tolist()
#
# for pre , label in zip(res['label'],df['label']):
#     if pre==label: correct+=1
#
# acc=correct/df['label'].shape[0]
# report = classification_report(y_label,y_pre,digits=4)
# print(f"acc:{acc:.4f}")
# print(f"report:\n{report}")
