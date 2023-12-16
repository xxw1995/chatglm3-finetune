from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from utils import process_data, DFaiss, Dprompt, test_bge_cos

class IntentAgent(BaseSingleActionAgent):
    tools: List
    llm: BaseLanguageModel
    intent_template: str = """
    有一些参考资料，为:{docs}
    你的任务是根据「参考资料」来理解用户问题的意图，并判断该问题属于哪一类意图。
    注意：你输出的意图应该是概率最高的那一个，不能是多个
    用户问题：“{query}”
    """

    prompt = PromptTemplate.from_template(intent_template)
    llm_chain: LLMChain = None

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def choose_tools(self, query):
        self.get_llm_chain()
        tool_names = [tool.name for tool in self.tools]

        if len(query) > 512:
            prompt_model = Dprompt()
            prompt_model.load_data("./doc/")
            retrive_knowledge = prompt_model.answer(query)
        else:
            # 知识库
            retrive_knowledge = test_bge_cos(query, "./doc/doc.txt",  mode="single")
            print("retrive_knowledge:", retrive_knowledge)
        resp = self.llm_chain.predict(query=query, docs=retrive_knowledge)
        select_tools = [(name, resp.index(name)) for name in tool_names if name in resp]
        select_tools.sort(key=lambda x:x[1])
        if len(select_tools) == 0:
            return ["default"]
        return [x[0] for x in select_tools]

    @property
    def input_keys(self):
        return ["input"]

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        tool_name = self.choose_tools(kwargs["input"])[0]
        return AgentAction(tool=tool_name, tool_input=kwargs["input"], log="")

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        raise NotImplementedError("IntentAgent does not support async")
