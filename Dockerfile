FROM python:3.10-buster

# copy the content to the target container
COPY . app
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

CMD ["streamlit", "run", "app.py"]
