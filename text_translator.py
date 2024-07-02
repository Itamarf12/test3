import starlette
import boto3
import os

#from transformers import pipeline

from ray import serve
import logging

ray_serve_logger = logging.getLogger("ray.serve")


@serve.deployment
class Translator:
    def __init__(self):
        #self.model = pipeline("translation_en_to_de", model="t5-small")

        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        print("11111111111111111111111")

        region = 'us-east-1'
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region
        )
        s3 = session.client('s3')
        bucket = 'nonsensitive-data'
        prefix = 'demo'
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            self.folders = [item['Key'] for item in response['Contents']]

    def translate(self, text: str) -> str:
        #return self.model(text)[0]["translation_text"]
        print("22222222222222222222")
        ray_serve_logger.warning("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):
        #req = await req.json()
        #return self.translate(req["text"])
        return self.folders



app = Translator.options(route_prefix="/translate").bind()


