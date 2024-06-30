import starlette

#from transformers import pipeline

from ray import serve


@serve.deployment
class Translator:
    def __init__(self):
        #self.model = pipeline("translation_en_to_de", model="t5-small")
        pass

    def translate(self, text: str) -> str:
        #return self.model(text)[0]["translation_text"]
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):
        #req = await req.json()
        #return self.translate(req["text"])
        return "aaaaaaaaaaaaaaaaaa"


app = Translator.options(route_prefix="/translate").bind()


