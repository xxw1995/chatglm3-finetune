from pypdf import PdfReader
import docx
import os
import faiss
import numpy as np
from transformers import AutoModel,AutoTokenizer
from sentence_transformers import SentenceTransformer

def process_data(root_path):
    all_content = []
    files = os.listdir(root_path)
    for file in files:
        path = os.path.join(root_path,file)
        if path.endswith(".docx"):
            doc = docx.Document(path)
            paragraphs = doc.paragraphs
            content = [i.text  for i in paragraphs]
            texts = ""
            for text in content:
                if len(text) <= 1:
                    continue
                if len(texts) > 150:
                    all_content.append(texts)
                    texts = ""
                texts += text
        elif path.endswith(".pdf"):
            with open(path,"rb") as f:
                pdf_reader = PdfReader(f)

                pages_info = pdf_reader.pages

                for page_info in pages_info:
                    text = page_info.extract_text()
                    all_content.append(text)
        elif path.endswith(".txt"):
            with open(path,encoding="utf-8") as f:
                lines = f.readlines()
                for content in lines:
                    all_content.append(content)
    return all_content

class DFaiss:
    def __init__(self):
        self.index = faiss.IndexFlatL2(4096)
        self.text_str_list = []

    def search(self, emb):
        D,I = self.index.search(emb.astype(np.float32), 100000)
        distance = 10000000
        if D[0][0] > distance:
            content = ""
        else:
            content = ""
            for i in range(len(self.text_str_list)):
                if D[0][i] < distance:
                    print("distance:", D[0][i])
                    print("str:", self.text_str_list[I[0][i]])
                    content += self.text_str_list[I[0][i]]
        return content

class Dprompt:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./model_base", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("./model_base", trust_remote_code=True).half().cuda()
        self.myfaiss = DFaiss()

    def answer(self, text):
        emb = self.get_sentence_emb(text,is_numpy=True)
        ans = self.myfaiss.search(emb)
        return ans
    def load_data(self,path):
        all_content = process_data(path)
        for content in all_content:
            self.myfaiss.text_str_list.append(content)
            emb = self.get_sentence_emb(content,is_numpy=True)
            self.myfaiss.index.add(emb.astype(np.float32))

    def get_sentence_emb(self,text,is_numpy=False):
        idx = self.tokenizer([text],return_tensors="pt")
        idx = idx["input_ids"].to("cuda")
        emb = self.model.transformer(idx,return_dict=False)[0]
        emb = emb.transpose(0,1)
        emb = emb[:,-1]

        if is_numpy:
            emb = emb.detach().cpu().numpy()

        return emb

def test_bge():
    from sentence_transformers import SentenceTransformer
    import csv
    queries = ["介绍一下胡桃", "介绍一下纲手", "介绍一下哈利波特"]
    passages = []
    data_path = "/data/npc_data.csv"
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        total_string_list = []
        for row in csv_reader:
            row_string = " ".join(row)
            passages.append(row_string)

    #print("passages:", passages, len(passages))

    instruction = "为这个句子生成表示以用于检索相关文章："
    model = SentenceTransformer('./models/bge-large-zh') # bge模型路径
    q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
    p_embeddings = model.encode(passages, normalize_embeddings=True)
    scores = q_embeddings @ p_embeddings.T
    print(scores)

def test_bge_cos(query, knowledge_file, mode):
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # 读取知识库文件并创建knowledge列表
    with open(knowledge_file, 'r', encoding="utf8") as f:
        knowledge_list = f.read().split('\n')

    # 加载模型
    model_path = './models/bge-large-zh' # bge模型路径
    model_cn = SentenceTransformer(model_path)

    knowledge_embeddings = model_cn.encode(knowledge_list, normalize_embeddings=True)

    instruction = "为这个句子生成表示，用于检索相关文章：" # 遵循bge的finetune模式
    query_embedding = model_cn.encode(f"{instruction}{query}", normalize_embeddings=True)

    similarities = cosine_similarity([query_embedding], knowledge_embeddings).flatten()
    print("similarities:", similarities)

    top_indices = np.argsort(-similarities)
    print("top_indices:", top_indices)

    def get_over_string(threshold=0.6):  # 相似度阈值作为参数
        if mode == "multi" and any(similarity > threshold for similarity in similarities):
            count = 1
            result_list = []
            for i in top_indices:
                if similarities[i] > threshold:
                    result_list.append(f"背景知识{count}:{knowledge_list[i]}")
                    count += 1
            return "\n".join(result_list)
        return "如果用户的问题意图不在参考资料中，则意图返回「default」"

    if mode == "single": # 只接受单一返回值 -> 用作意图判断，不接收多意图
        return (knowledge_list[top_indices[0]] if similarities[top_indices[0]] >= 0.6
                else "如果用户的问题意图不明确，则意图返回「default」")
    elif mode == "multi": # 接受多返回值 -> 用作数据召回
        return get_over_string()
