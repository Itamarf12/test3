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

def download_directory_from_s3(access_key, secret_key, region, bucket_name, s3_directory, local_directory):
    ray_serve_logger.warning("Start Model downloading ..")
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
        ray_serve_logger.warning(f"Model downloaded {obj.key} to {target}")


def load_model(model_path):
    ray_serve_logger.warning("Start Model loading ..")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    compute_dtype = torch.float32
    device = torch.device(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        return_dict=False,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True
    )
    ray_serve_logger.warning(f"Model was loaded successfully.")
    return model, tokenizer


def get_next_word_probabilities(sentence, tokenizer, device, model, top_k=1):

    # Get the model predictions for the sentence.
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(device)  # .cuda()
    outputs = model(inputs)
    predictions = outputs[0]


    # Get the next token candidates.
    next_token_candidates_tensor = predictions[0, -1, :]

    # Get the top k next token candidates.
    topk_candidates_indexes = torch.topk(
        next_token_candidates_tensor, top_k).indices.tolist()

    # Get the token probabilities for all candidates.
    all_candidates_probabilities = torch.nn.functional.softmax(
        next_token_candidates_tensor, dim=-1)

    # Filter the token probabilities for the top k candidates.
    topk_candidates_probabilities = \
        all_candidates_probabilities[topk_candidates_indexes].tolist()

    # Decode the top k candidates back to words.
    topk_candidates_tokens = \
        [tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

    # Return the top k candidates and their probabilities.
    return list(zip(topk_candidates_tokens, topk_candidates_probabilities))


def get_first_line(file_path):
    """Reads and returns the first line of a file."""
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            return first_line
    except FileNotFoundError:
        return f"Error: The file {file_path} does not exist."
    except Exception as e:
        return f"Error: An unexpected error occurred. {str(e)}"



def download_directory(bucket_name, source_directory, destination_directory):
    """Downloads all files from a specified directory in a bucket to a local directory."""
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List all blobs in the directory
    blobs = bucket.list_blobs(prefix=source_directory)

    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for blob in blobs:
        # Determine the local file path
        local_path = os.path.join(destination_directory, os.path.relpath(blob.name, source_directory))
        local_dir = os.path.dirname(local_path)

        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Download the file
        if source_directory != blob.name:
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")


def get_risky_score(sentence, tokenizer, device, model):
    res = get_next_word_probabilities(sentence, tokenizer, device, model, top_k=1)
    choosen_res = res[0]
    return choosen_res[1] if choosen_res[0].lower()=='pos' else 1-choosen_res[1]



@serve.deployment
class Translator:
    def __init__(self):
        self.device = DEVICE
        # aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        # aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        # ray_serve_logger.warning("aaaaaaaaaaaaaaa  1111111")
        # encoded_key = os.getenv('GCP_CRED')
        # ray_serve_logger.warning(f"aaaaaaaaaaaaaaa   22222   {encoded_key}")
        # decoded_key = base64.b64decode(encoded_key).decode('utf-8')
        # ray_serve_logger.warning(f"aaaaaaaaaaaaaaa 33333   {decoded_key}")
        # with open('/tmp/temp_credentials.json', 'w') as temp_file:
        #     temp_file.write(decoded_key)
        # ray_serve_logger.warning(f"aaaaaaaaaaaaaaa 4444444")
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/tmp/temp_credentials.json'
        #
        # bucket_name = "apiiro-trained-models"  # "your-bucket-name"
        # source_directory = "risky-feature-requests/phi-3/"  # "path/to/your/source-file"
        # destination_directory = MODEL_LOCAL_DIR

        #download_directory(bucket_name, source_directory, destination_directory)

        #download_directory_from_s3(aws_access_key_id, aws_secret_access_key, REGION, BUCKET, S3_DIRECTORY, MODEL_LOCAL_DIR)
        #self.model, self.tokenizer = load_model(MODEL_LOCAL_DIR)

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


