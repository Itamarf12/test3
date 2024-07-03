import starlette
import boto3
import os
import random
import torch
from transformers import pipeline, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from ray import serve
import logging

ray_serve_logger = logging.getLogger("ray.serve")



def download_directory_from_s3(access_key, secret_key, region, bucket_name, s3_directory, local_directory):
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    # Ensure the local directory exists
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # Iterate over objects in the S3 directory
    for obj in bucket.objects.filter(Prefix=s3_directory):
        target = os.path.join(local_directory, os.path.relpath(obj.key, s3_directory))

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        if obj.key.endswith('/'):
            continue  # Skip directories, only download files

        bucket.download_file(obj.key, target)
        print(f"Downloaded {obj.key} to {target}")

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    finetuned_model = "/Users/itamarlevi/Downloads/phi_model_2"
    compute_dtype = torch.float32
    device = torch.device("cpu")
    model = AutoPeftModelForCausalLM.from_pretrained(
        finetuned_model,
        torch_dtype=compute_dtype,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer


@serve.deployment
class Translator:
    def __init__(self):
        #self.model = pipeline("translation_en_to_de", model="t5-small")

        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = 'us-east-1'
        bucket_name = 'nonsensitive-data'
        s3_directory = 'phi3-small'
        local_directory = '/tmp/phi3'
        #os.makedirs(local_directory)
        download_directory_from_s3(aws_access_key_id, aws_secret_access_key, region, bucket_name, s3_directory, local_directory)


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
        return "bbbbbbbbbbbb"

    async def __call__(self, req: starlette.requests.Request):
        print("3333333333333333")
        ray_serve_logger.warning("r1rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        file_path = f"/tmp/text_{random.randint(1, 10)}.txt"
        with open(file_path, 'w') as file:
            file.write('Hello, world!')

        ray_serve_logger.warning("r2rrrrrrrrrrrrrrrwrote rrrrrrrrrrrrrrrrrrrrrrrr")
        current_path = os.getcwd()
        #current_path = os.path.abspath(__file__)
        ray_serve_logger.warning("r3rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        ray_serve_logger.warning(f"kkkkkkkkkkkkkkkkkkkkkkkkkCurrent Path:   {current_path}")
        #req = await req.json()
        #return self.translate(req["text"])
        return self.folders



app = Translator.options(route_prefix="/translate").bind()


