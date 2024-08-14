#     docker build . -t build_chatbot_for_your_data_model
#     docker run -p 8000:8000 build_chatbot_for_your_data_model
#(Need to run every time changes are made)
FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "-u", "server.py"]