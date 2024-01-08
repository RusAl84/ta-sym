from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
import os
import time
import urllib.request
import process_nlp


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def dafault_route():
    return 'API'


@app.route('/uploadae', methods=['POST'])
@cross_origin()
def uploadae():
    for fname in request.files:
        f = request.files.get(fname)
        print(f)
        milliseconds = int(time.time() * 1000)
        filename = str(milliseconds)
        # f.save('./uploads/%s' % secure_filename(fname))
        full_filename = f"./uploads/{milliseconds}.json"
        f.save(full_filename)
        text = process_nlp.convertJsonMessages2text(full_filename)
        d = {}
        d['text'] = text
        d['filename'] = filename
    return d


@app.route('/uploadaem', methods=['POST'])
@cross_origin()
def uploadaem():
    for fname in request.files:
        f = request.files.get(fname)
        print(f)
        milliseconds = int(time.time() * 1000)
        filename = str(milliseconds)
        # # f.save('./uploads/%s' % secure_filename(fname))
        # full_filename = f"./uploads/{milliseconds}.json"
        # f.save(full_filename)
        # text = process_nlp.convertJsonMessages2text(full_filename)
        d = {}
        # d['text'] = text
        # d['filename'] = filename
    return d

@app.route("/get_pattern", methods=['POST'])
def get_pattern():
    msg = request.json
    print(msg)
    data = process_nlp.get_pattern(msg['text'])
    data = process_nlp.add_print_text(data)
    print(data['print_text'])
    return data


@app.route("/get_pattern_add", methods=['POST'])
def get_pattern_add():
    msg = request.json
    print(msg)
    data = process_nlp.add_data(msg['text'])
    data = process_nlp.add_print_text(data)
    print()
    print(data)
    return data


@app.route('/findae', methods=['POST'])
def findae():
    #     if request.method == 'POST':
    msg = request.json
    print(msg)
    filename = msg['filename']
    counts=msg['counts']
    text=msg['text']
    # filename="d:/ml/chat/andromedica1.json"
    save_filename="./data_proc.json"
    # data_proc(filename, save_filename, 32)
    # find_cl(save_filename)
    data = process_nlp.find_cl_text(text, counts)
    print(data)
    return data


@app.route("/clear_db", methods=['GET'])
def clear_db():
    process_nlp.clear_db()
    return "ok clear_db"


@app.route("/load_db", methods=['GET'])
def load_db():
    data = process_nlp.load_db()
    return data


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
# app.run(host="0.0.0.0")