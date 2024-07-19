from ray import serve
import starlette
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
DEVICE = 'gpu'

MODEL_NM = "Open-Orca/Mistral-7B-OpenOrca"
#MODEL_NM = "microsoft/Orca-2-13b"




# def get_prompt1(title, description):
#     prompt_prefix = """
# This ticket is classified as high-risk. Based on the context provided, choose the most appropriate risk category and give a concise reason. Respond with just the selected category and its one-sentence justification, in the following format: "Risk Category: [Selected Category], Reason: [Brief Explanation]."
# Risk Categories:
# - Sensitive Data Handling
# - User Access and Identity Management
# - APIs and Web Services
# - External Applications and Integrations
# - Infrastructure and Platform Security
# If none apply, use "Other" for the category.
#     """
#     return f"""
# This is the details of a ticket:
# title:
# {title}
# description:
# {description}
# {prompt_prefix}
#     """
#
#
# def categorical_response1(model, title, description):
#     system_message = """
# Role: Application Security (AppSec) Assistant
# Directive: Adhere strictly to the provided guidelines.
# Task: Upon review of the specified Jira ticket, determine and concisely state the security risk it presents.
#     """
#     user_message = get_prompt1(title, description)
#     prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
#     inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
#     outputs = model.generate(
#     **inputs, max_new_tokens=256, use_cache=True, do_sample=True,
#     temperature=0.2, top_p=0.95)
#     res = tokenizer.batch_decode([outputs[0][inputs['input_ids'].size(1):]])[0]
#     return res
#
#
#


@serve.deployment
class RiskyReasoning:
    def __init__(self):
        self.device = DEVICE


    def translate(self, text: str) -> str:
        #return self.model(text)[0]["translation_text"]
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):

        req = await req.json()
        re = 'NO DATA - missing text field'
       # model = AutoModelForCausalLM.from_pretrained(MODEL_NM)
        #tokenizer = AutoTokenizer.from_pretrained(MODEL_NM)

        return re


#app = Translator.options(route_prefix="/translate").bind()
app = RiskyReasoning.bind()
