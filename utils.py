from pypdf import PdfReader
import docx
import os
import faiss
import numpy as np
from transformers import AutoModel,AutoTokenizer

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
        self.tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("./model", trust_remote_code=True).half().cuda()
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
