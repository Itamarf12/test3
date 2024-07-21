import starlette
from transformers import AutoTokenizer, AutoModelForCausalLM
from ray import serve
import logging
from google.cloud import storage

ray_serve_logger = logging.getLogger("ray.serve")
MODEL = "Open-Orca/Mistral-7B-OpenOrca"
#MODEL = 'microsoft/DialoGPT-small'
DEVICE = 'cpu'


@serve.deployment
class RiskyReasoning:
    def __init__(self):
        self.device = DEVICE

    def translate(self, text: str) -> str:
        #return self.model(text)[0]["translation_text"]
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL)

        req = await req.json()
        re = 'NO DATA - missing text field'

        ray_serve_logger.warning(f"Missing text field in the json  request = {req}")
        return re


#app = Translator.options(route_prefix="/translate").bind()
app = RiskyReasoning.bind()


