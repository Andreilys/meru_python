# Meru: "Fine-Tune Generative AI for Your Application"

Meru is a scalable API powered by Dreambooth and Stable diffusion that brings personalized, user-specific content generation to any application. With Meru, you can develop and deploy fine-tuned stable diffusion models in minutes without provisioning for compute.

You can read Meru docs here - https://usemeru.com/docs

*Sample output*
**Prompt: a sks person jedi with short hair standing still looking at the sunset concept art by Doug Chiang cinematic, realistic painting, high definition, concept art, portait image, path tracing, serene landscape, high quality, highly detailed, 8K, soft colors, warm colors, turbulent sea, high coherence, anatomically correct, hyperrealistic, concept art, defined face, five fingers, symmetrical**
[](jedi_andrei.jpeg)


# Instructions:
0) git clone this repo
1) Make sure you have at least 3 credits from https://usemeru.com/credits
2) Set meru_api_key environment variable to your api key or enter it when initializing MeruApp()
	1) For example export meru_api_key=YOUR_API_KEY in command line
	2) or os.environ['meru_api_key']=YOUR_API_KEY in Python
4) Upload 5-20 images of your subject in the /input/ folder 
5) Run *ipython -i meru_app.py*
	1) This starts ipython with the MeruApp class 
6) Initialize the class (m = MeruApp())
	1) When picking a class, choose a name like person, dog, cat that generalizes for what you're training on
	2) The script will save the train_id/class to model_details.txt for future reference the first time you run it 
	3) if loading a previous model, pass in meruApp(load_model=True)
	4) This will read from model_details.txt which contains your class name and train id
7) Call m.train() to train your model
8) Once trained you can download it to the img_model directory with m.download_model()
9) Run your prompt with prompt() or run multiple prompts with prompt_file() 
	1) make sure your prompt contains *sks {CLASS_NAME}* or it wont run.
	2) Update prompts.txt with your prompts if you want to run through different prompts
	3) Adjust sample size with num_samples (i.e. m.prompt(num_samples=6))
10) After running, your outputs will be saved in /output/, make sure you save them elsewhere before running again since they will be overwritten.
