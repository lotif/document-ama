# Document AMA

Building upon [kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference](https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference) with a Streamlit
front-end. It allows to load any `.txt` file or `.pdf` document with text and asking any questions about
it using LLMs on CPU.

![Screenshot of the app](reference_image.png?raw=true "Screenshot of the app")

## Setting up

### Docker

Build the docker image:
```shell
docker build -t document-ama .
```

Run it:
```shell
docker run -d -p 8501:8501 --name document-ama document-ama:latest
```

Open the Streamlit app in your browser:
```shell
http://localhost:8502/
```

### Local env

Init a VENV, if desired:
```shell
python3 -m venv venv
source venv/bin/activate
```

Install requirements:
```sh
pip install -r requirements.txt
```

Run streamlit:
```shell
streamlit run app.py
```

A new browser window will pop up with the streamlit app.
