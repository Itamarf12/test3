import starlette
import boto3
import os
import random
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from ray import serve
import logging
import base64
import json
from google.cloud import storage

ray_serve_logger = logging.getLogger("ray.serve")
BUCKET = 'nonsensitive-data'
REGION = 'us-east-1'
S3_DIRECTORY = 'phi3_finetuned'
MODEL_LOCAL_DIR = '/tmp/phi3'
DEVICE = 'cpu'


@serve.deployment
class Translator:
    def __init__(self):
        self.device = DEVICE

    def translate(self, text: str) -> str:
        #return self.model(text)[0]["translation_text"]
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):

        req = await req.json()
        re = 'NO DATA - missing text field'
        if 'text' in req:
            sentence = req['text']
            #re = get_next_word_probabilities(sentence, self.tokenizer, self.device, self.model, top_k=2)
            re = get_risky_score(sentence, self.tokenizer, DEVICE, self.model)
        else:
            ray_serve_logger.warning(f"Missing text field in the json  request = {req}")
        return re


#app = Translator.options(route_prefix="/translate").bind()
app = Translator.bind()

