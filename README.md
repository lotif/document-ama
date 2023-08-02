# Document AMA

Building upon [kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference](https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference) with a Streamlit
front-end. It allows to load any `.txt` file or `.pdf` document with text and asking any questions about
it using LLMs on CPU.

## Setting up

### Docker

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
