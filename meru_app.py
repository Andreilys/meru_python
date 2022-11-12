import requests
import os
from os import listdir, environ
from os.path import isfile, join
import time
from PIL import Image
import imghdr
import urllib

class MeruApp():
    def __init__(self, input_img_file_path='input', output_img_file_path='output', load_model=False):
        '''
        Class object takes in input/output img file paths.
        Make sure you have at least 3 credits here: https://usemeru.com/credits.
        You can pass in prev_model=True to initialize an old model. 
        Make sure model_details.txt contains trian_id and class_name, for example:
            train_id:xxxx
            class_name:person
        '''
        if environ.get('meru_api_key') is not None:
            api_key = environ.get('meru_api_key')
        else:
            api_key = input('Please enter your API key: ')        
            os.environ["meru_api_key"] = api_key 
        self.HEADERS = {
                'x-api-key': api_key,
        }
        # Preload an exisiting train_id
        if load_model:
            with open('model_details.txt') as file:
                lines = [line.rstrip() for line in file]
            self.TRAIN_ID = lines[0].split(':')[1]
            # you've already trained the model previously on your class name
            self.CLASS = lines[1].split(':')[1]
        else:
            self.CLASS = input('Please enter your class name (i.e. person, dog, cat): ') 
            self.TRAIN_ID = requests.post('https://api.usemeru.com/train/create', headers=self.HEADERS).json()['train_id']
            # write train_id and class_name to file in case you need it later
            with open("model_details.txt", "w") as text_file:
                text_file.write(f'train_id:{self.TRAIN_ID}\n')
                text_file.write(f'class_name:{self.CLASS}')
        self.TRAIN_ID_FILES = {
            'train_id': self.TRAIN_ID,
        }
        self.INPUT_IMAGE_FILE_PATH = input_img_file_path
        self.OUTPUT_IMAGE_FILE_PATH = output_img_file_path
    

    def upload_images(self):
        '''
            The upload_images function uploads the images stored in the input_image_file_path to the Meru servers for training. 
            Currently the image types supported are .jpg, .jpeg, and .png
        '''
        onlyfiles = [f for f in listdir(self.INPUT_IMAGE_FILE_PATH) if isfile(join(self.INPUT_IMAGE_FILE_PATH, f))]
        if len(onlyfiles) == 0:
            raise ValueError(f"Make sure you have image files stored in {self.INPUT_IMAGE_FILE_PATH}")
        for img in onlyfiles:
            img_path = f'{self.INPUT_IMAGE_FILE_PATH}/{img}'
            img_type = imghdr.what(img_path)
            # jpg, png only
            if img_type in ('jpg', 'jpeg', 'png'):
                print(f'Posting: {img_path}')
                files = {
                    'train_id': self.TRAIN_ID,
                    'images': open(f'{img_path}', 'rb'),
                }
                requests.post('https://api.usemeru.com/train/upload', headers=self.HEADERS, files=files)
        print('Uploaded images')


    def train(self, epochs=1600, callback_url=None, text_enc=100, download_model=True, download_model_dir='img_model'):
        '''
            The train function takes in epochs, callback_url and text encoder and trains the model.
            Epochs can be as high as 4000.
            If you don't want to download the model, set download_model to false
        '''
        self.upload_images()
        files = {
            'train_id': self.TRAIN_ID, 
            'class': self.CLASS,
            'epochs' : str(epochs),
            'text_enc' : str(text_enc),
            'callback' : callback_url
        } 
        requests.post('https://api.usemeru.com/train/run', headers=self.HEADERS, files=files)
        start = time.time()
        print(f'Starting to train with id: {self.TRAIN_ID}')
        # Loop until model finishes training
        while True:
            response = requests.post('https://api.usemeru.com/train/status', headers=self.HEADERS, files=self.TRAIN_ID_FILES)
            if 'model_uri' in response.json().keys():
                end = time.time()
                print(f'Model Trained. Minutes took to train: {(end - start)/60}')
                break
            end = time.time()
            print(f'Model training, Minutes Elapsed: {(end - start)/60}')
            time.sleep(180)
        if download_model:
            self.download_model(download_model_dir)


    def open_images(self):
        '''
            the open_images() gets all the images from the OUTPUT_IMAGE_FILE_PATH and combines them into one single image that it opens 
        '''
        files = [f for f in listdir(self.OUTPUT_IMAGE_FILE_PATH) if isfile(join(self.OUTPUT_IMAGE_FILE_PATH, f))]
        images = [Image.open(f'{self.OUTPUT_IMAGE_FILE_PATH}/{x}') for x in files]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        save_file = f'{self.OUTPUT_IMAGE_FILE_PATH}/combined_img_output/combined_img_output.jpg'
        new_im.save(save_file)
        im = Image.open(save_file)
        im.show()


    def get_images(self, prompt, num_samples=1, seed=42, guidance_scale=10, callback_url=None):
        '''
            the get_images() function takes in a prompt, seed, guidance scale and a number of samples, saving the images to your OUTPUT_IMAGE_FILE_PATH.
            This will over-write previous images so make sure you move them if you want to save them!
        '''
        if self.CLASS not in prompt:
            raise ValueError(f'Please make sure your class name: {self.CLASS} is in the prompt')
        print(f'Running inference on: {prompt}')
        files = {
            'train_id': self.TRAIN_ID,
            'prompt': prompt,
            'num_samples' : num_samples,
            'seed' : str(seed),
            'guidance_scale' : int(guidance_scale),
            'callback': callback_url
        }
        response = requests.post('https://api.usemeru.com/inference/infer', headers=self.HEADERS, files=files)
        infer_id = response.json()['infer_id']
        files = {
            'infer_id': infer_id,
        }
        start = time.time()
        while True:
            response = requests.post('https://api.usemeru.com/inference/status', headers=self.HEADERS, files=files)
            if 'infer_uris' in response.json().keys():
                end = time.time()
                print(f'Inference Finished. Seconds took to run: {(end - start)}')
                break
            end = time.time()
            print(f'Running inference, seconds elapsed: {(end - start)}')
            time.sleep(40)
        img_uri = response.json()['infer_uris'] 
        print(f'Saving images to: {self.OUTPUT_IMAGE_FILE_PATH}')
        # save images
        for idx, img_urls in enumerate(img_uri):
            img_data = requests.get(img_urls).content
            img_fn = f'{self.OUTPUT_IMAGE_FILE_PATH}/output_img_{idx}.jpg' 
            with open(img_fn, 'wb') as handler:
                handler.write(img_data) 
        print(f'Images saved to: {self.OUTPUT_IMAGE_FILE_PATH}. Make sure you move them before running prompt() again or theyll be overwritten.')


    def prompt(self, prompt, num_samples=1, seed=42, guidance_scale=10, callback_url=None):
        '''
            the prompt() function takes in a prompt, seed, guidance scale and a number of samples, saving the images to your OUTPUT_IMAGE_FILE_PATH.
            This will over-write previous images so make sure you move them if you want to save them!
        '''
        self.get_images(prompt, num_samples, seed, guidance_scale, callback_url)
        self.open_images()


    def prompt_file(self, file_path='prompts.txt', num_samples=1, seed=42, guidance_scale=10, callback_url=None):
        '''
            the prompt_file() function takes in a prompt file path, seed, guidance scale and a number of samples, saving the images to your OUTPUT_IMAGE_FILE_PATH.
            This will over-write previous images so make sure you move them if you want to save them!
        '''
        with open(file_path) as file:
            lines = [line.rstrip() for line in file]
        for prompt in lines:
            self.prompt(prompt, num_samples, seed, guidance_scale, callback_url)
            print('----------------------------------------------------------------')
            print('----------------------------------------------------------------')


    def download_model(self, output_dir='img_model'):
        '''
            the download_model() function takes in a output directory and saves the model as ckpt file
        '''
        response = requests.post('https://api.usemeru.com/train/status', headers=self.HEADERS, files=self.TRAIN_ID_FILES) 
        url = response.json()['model_uri']
        urllib.request.urlretrieve(url, filename=f"{output_dir}/{CLASS}_img_model.ckpt")


    def delete_model(self):
        '''
            the delete_model() function deletes the model for the current train_id 
        '''
        print(f'Deleting model with train_id: {self.TRAIN_ID}')
        response = requests.post('https://api.usemeru.com/train/delete', headers=self.HEADERS, files=self.TRAIN_ID_FILES)
        return response.json()


