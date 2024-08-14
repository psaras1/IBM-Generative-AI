# Flask is a lightweight, modular Python web framework that enables rapid
# development of web applications with minimal setup and flexibility to add
# desired features.
from flask import Flask

app = Flask(__name__)


# Specify URL path
@app.route('/')
def hello_world():
  return 'Hello, World!'


@app.route('/test')
def hello_test():
  return 'Hello, Test!'


# Debug = True will automatically reload the server when changes are made
if __name__ == '__main__':
  app.run(debug=True)
