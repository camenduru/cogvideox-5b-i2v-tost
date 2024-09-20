import os, json, requests, random, runpod

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel, T5Tokenizer

with torch.inference_mode():
    model_id = "/content/model"
    transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae, torch_dtype=torch.float16).to("cuda")
    # pipe.enable_model_cpu_offload()

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    original_file_name = url.split('/')[-1]
    _, original_file_extension = os.path.splitext(original_file_name)
    file_path = os.path.join(save_dir, file_name + original_file_extension)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image_check']
    input_image = download_file(url=input_image, save_dir='/content/input', file_name='input_image_tost')
    prompt = values['prompt']
    # guidance_scale = values['guidance_scale']
    # use_dynamic_cfg = values['use_dynamic_cfg']
    # num_inference_steps = values['num_inference_steps']
    # fps = values['fps']
    guidance_scale = 6
    use_dynamic_cfg = True
    num_inference_steps = 50
    fps = 8

    image = load_image(input_image)
    video = pipe(image=image, prompt=prompt, guidance_scale=guidance_scale, use_dynamic_cfg=use_dynamic_cfg, num_inference_steps=num_inference_steps).frames[0]
    export_to_video(video, "/content/cogvideox_5b_i2v_tost.mp4", fps=fps)

    result = "/content/cogvideox_5b_i2v_tost.mp4"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)

runpod.serverless.start({"handler": generate})