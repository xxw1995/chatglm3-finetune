import requests
import json
import io
import base64
import os
from PIL import Image
from typing import Optional

from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
import re, random
from hashlib import md5
from utils import test_bge_cos
from self_tools import system_info

class APITool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""

    def _call_api(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_api(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")

class functional_Tool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""

    def _call_func(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_func(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")

# search tool #
class SearchTool(APITool):
    llm: BaseLanguageModel

    name = "搜索问答"
    description = "根据用户问题搜索最新的结果，并返回Json格式的结果"

    # search params
    google_api_key: str
    google_cse_id: str
    url = "https://www.googleapis.com/customsearch/v1"
    top_k = 2

    # QA params
    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    通过Search，如果该文本中没有相关内容可以回答问题，请直接回复：“抱歉，通过Search该问题需要更多上下文信息。”
    ```{text}```
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_api(self, query):
        self.get_llm_chain()
        context = self.get_search_result(query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_search_result(self, query):
        data = {"key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "lr": "lang_zh-CN"}
        results = requests.get(self.url, params=data).json()
        results = results.get("items", [])[:self.top_k]
        snippets = []
        if len(results) == 0:
            return("No Search Result was found")
        for result in results:
            print("result:", result)
            text = ""
            if "title" in result:
                text += result["title"] + "。"
            if "snippet" in result:
                text += result["snippet"]
            snippets.append(text)
        return("\n\n".join(snippets))

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

class Text_classification_Tool(functional_Tool):
    llm: BaseLanguageModel

    name = "文本分类"
    description = "用户输入句子，完成文本分类"

    # QA params
    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    ```{text}```
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = "Instruction: 你是一个非常厉害的[词条名称]多层级分类模型"
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

class Default_Tool(functional_Tool):
    llm: BaseLanguageModel
    name = "default"
    description = "默认对话工具，根据用户输入完成对话"

    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    ```{text}```
    问题：{query}
    """

    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = "Instruction: 你是一个聊天机器人，需要根据用户的输入完成聊天任务，你需要根据下面这些背景知识来回答："
        # 知识库
        retrive_knowledge = test_bge_cos(query, "data/npc_data.csv", mode="multi")
        # retrive_knowledge不为空，则调用base_model，结合知识库输出内容
        if retrive_knowledge != "":
            context += retrive_knowledge
            resp = self.llm_chain.predict(text=context, query=query)
            return resp
        # retrive_knowledge为空，则代表没有相似的知识，用对齐后的chat_model生成知识，再输出内容
        # chat模型可以结合定义好的tools
        else:
            model_response, model_history=self.model_chat(query)
            context += model_response
            resp = self.llm_chain.predict(text=context, query=query)
            return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def model_chat(self, task_query):
        from transformers import AutoTokenizer, AutoModel
        base_dir = "models/chatglm3-6b"
        tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(base_dir, trust_remote_code=True).cuda()
        model = model.eval()
        model_history = [system_info]
        model_response, model_history = model.chat(tokenizer, task_query, history=model_history)
        return self.run_task(model_response, model_history)

    def run_task(self, model_response, model_history):
        if isinstance(model_response, dict):
            print("model_response is dict")
            import function_map
            # 使用 getattr 来从 function_map 中获取 model_response 字典中 "name" 键对应的函数。
            func = getattr(function_map, model_response.get("name"))
            param = model_response.get("parameters")
            func_response = func(**param)
            result = json.dumps(func_response, ensure_ascii=False)
            # 把前一次的输出放入，再次调用
            model_response, model_history = model.chat(tokenizer, result, history=model_history, role="observation")
            # 再次执行run_task
            model_response, model_history = run_task(model_response, model_history)
            return model_response, model_history
        else:
            return model_response, model_history
