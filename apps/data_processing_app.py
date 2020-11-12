#!/gpfs/mskmindhdp_emc/sw/env/bin/python3
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, MSK!"

if __name__ == '__main__':
    app.run(host = os.environ['HOSTNAME'], debug=True)
