import os
from flask import (
    Flask,
    jsonify,
    render_template,
    flash,
    redirect,
    url_for,
    session,
    request,
    send_from_directory,
)
import json
# 初始化
app = Flask(__name__)

storage_data = []
data_one = []


# @app.route('/', methods=['GET', "POST"])
# def index():
#     return jsonify("///")



@app.route('/api/rec', methods=['POST'])
def rec():
    print(request.data)
    data = str(request.data, 'utf-8')
    jdata = json.loads(data)
    data_one.append(jdata)
    storage_data.append(jdata)
    print(jdata)
    return jsonify(200, jdata)


@app.route('/api/recent', methods=['GET'])
def recent():
    j = json.dumps(storage_data)
    print(j)
    return jsonify(j)


@app.route('/', methods=['GET'])
def recent_one():
    j = json.dumps(data_one[-1])
    print(j)
    return jsonify(j)


if __name__ == '__main__':
    config_run = dict(
        debug=True,
        host='127.0.0.1',
        port=8888,
    )
    app.run(**config_run)
