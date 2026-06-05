from prepare_tools import get_tools_dict,load_functions_from_file
from ollama import chat
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class AnalystAgent:
    def __init__(self,model_name="qwen3:4b",
        tools_path = "src/tools",
        data_path="data/ecommerce_customer_analytics.csv"
        ):
        self.model_name = model_name
        self.tools_path =tools_path
        self.data_path = data_path
        self.tools = []
        self.tools_dict = dict()
        self.get_tools()


    def get_tools(self):
        self.tools = []
        for file in os.listdir(self.tools_path):
            if file.endswith("py"):
                self.tools.extend(load_functions_from_file(os.path.join(self.tools_path,file)))

        self.tools_dict = dict()
        for file in os.listdir(self.tools_path):
            if file.endswith("py"):
                self.tools_dict={**self.tools_dict,**get_tools_dict(os.path.join(self.tools_path,file))}

    def answer_question(self):
        input_question = input("Enter your analysis question?")
        messages = [
            {"role":"system","content":"You are an AI assitance data analysis. Your role is to help answering and "
            "performing data analysis questions for a specific dataset"},
            {"role":"user","content":
             f"""
                Question:
                {input_question}

                Dataset path:
                {self.data_path}
                """},
        ]
        
        response = chat(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                stream=False
                
            )
            
        messages.append(response["message"])
        tool_calls =response["message"].get("tool_calls",[])
        for i in range(len(tool_calls)):
            try:
                    tool_name= tool_calls[i].function.name
                    if tool_name not in self.tools_dict:
                        result = f"Unkown tool: {tool_name}"
                    else:
                        result= self.tools_dict[tool_name](**tool_calls[i].function.arguments)
            except Exception as e:
                    result = f"Tool execution failed: {e}"
                
            messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result),
                    })
        response = chat(
                model=self.model_name,
                messages=messages,
                
            )
        return response['message']['content']

if __name__ =="__main__":
    agent = AnalystAgent(

    )

    answer = agent.answer_question()
    print(answer)