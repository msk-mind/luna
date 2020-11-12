#!/gpfs/mskmindhdp_emc/sw/env/bin/python3
from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, MSK!"

@app.route('/mind/api/v1/transfer', methods=['POST'])
def transfer():
    config = request.json
    return str(config) 

@app.route('/mind/api/v1/delta', methods=['POST'])
def delta():
    config = request.json
    return str(config) 

@app.route('/mind/api/v1/graph', methods=['POST'])
def graph():
    config = request.json
    return str(config) 

if __name__ == '__main__':
    app.run(host = os.environ['HOSTNAME'], debug=True)
