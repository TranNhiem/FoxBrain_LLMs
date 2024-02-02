'''
@Po-Kai 2023/12
A more convenient and user-friendly multilingual translation tool implemented using the OpenAI API.
'''
from .cores.base import OpenAIGenerator


class OpenAITaskSynthesizerPrompt(object):
    
    def __init__(self):
        self.prompt = self._generate_prompt()
    
    def _generate_prompt(self):
        return {
            "system": """您是一位樂於協助、尊重且誠實的助手。始終以盡可能有幫助的方式回答問題，同時確保安全。您的回答不應包含任何有害、不道德、種族主義、毒性、危險或非法的內容。如果某個問題毫無意義，或者在事實上不一致，請解釋原因，而不是提供不正確的回答。如果您不知道某個問題的答案，請不要分享虛假的資訊。""",
            
            "user": """
給定的以下文章摘錄涵蓋了廣泛的主題，您將生成多種用於訓練的自然語言處理（NLP）任務，並且每一種都給五個範例。這些任務應該涵蓋以下類別：

1. 自然語言推理（NLI）：創建一個任務，呈現文章中的前提，並詢問它是否包含給定的假設。格式為 "輸入：" {{model_response}}，"假設："，"輸出：" {{model_response}}。

2. 常識推理：創建一個任務，詢問文章中特定陳述背後的原因，需要應用常識知識。格式為 "輸入：" {{model_response}}，"輸出：" {{model_response}}。

3. 釋義檢測：創建一個任務，要求識別文章中是否有兩個陳述相互解釋。格式為 "片語1：" {{model_response}}，"片語2："，"輸出：" {{model_response}}。

4. 文章填充：創建一個任務，根據提供的信息問如何完成文章。格式為 "輸入：缺少部分的文本描述_______"，"輸出：" {{model_response}}。

開放和封閉型問答：從文章中創建開放式問題和具體答案的問題。格式為 "問題：" {{model_response}}，"答案：" {{model_response}}。

5. 閱讀理解：創建一個任務，呈現文章的一部分並提問以評估理解。格式為 "問題：" {{model_response}}，"答案：" {{model_response}}。

\n\n 這是文章摘錄：{article}。"""
        }
        

    def __call__(self, article):
        return [
            {"role": "system", "content": self.prompt["system"]},
            {"role": "user", "content": self.prompt["user"].format(article=article)},
        ]


class OpenAITaskSynthesizer(OpenAIGenerator):

    def _get_prompt_factory(self):
        return OpenAITaskSynthesizerPrompt()

    def generate(self, article):
        # In order to be compatible with existing codes.
        return self(article=article)
