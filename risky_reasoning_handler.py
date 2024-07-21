import starlette
from transformers import AutoTokenizer, AutoModelForCausalLM
from ray import serve
import logging
from google.cloud import storage

ray_serve_logger = logging.getLogger("ray.serve")
#MODEL = "Open-Orca/Mistral-7B-OpenOrca"
MODEL = "microsoft/Orca-2-13b"
#MODEL = 'microsoft/DialoGPT-small'
DEVICE = 'cpu'



def get_prompt1(title, description):
    prompt_prefix = """
This ticket is classified as high-risk. Based on the context provided, choose the most appropriate risk category and give a concise reason. Respond with just the selected category and its one-sentence justification, in the following format: "Risk Category: [Selected Category], Reason: [Brief Explanation]."
Risk Categories:
- Sensitive Data Handling
- User Access and Identity Management
- APIs and Web Services
- External Applications and Integrations
- Infrastructure and Platform Security
If none apply, use "Other" for the category.
    """
    return f"""
This is the details of a ticket:
title:
{title}
description:
{description}
{prompt_prefix}
    """


@serve.deployment
class RiskyReasoning:
    def __init__(self):
        self.device = DEVICE
        ray_serve_logger.warning(f"1111111111111")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        ray_serve_logger.warning(f"2222222222222")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL)
        ray_serve_logger.warning(f"3333333333")

    def translate(self, text: str) -> str:
        #return self.model(text)[0]["translation_text"]
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):

        req = await req.json()
        re = 'NO DATA - missing text field'

        ray_serve_logger.warning(f"Missing text field in the json  request = {req}")
        return re


#app = Translator.options(route_prefix="/translate").bind()
app = RiskyReasoning.bind()


