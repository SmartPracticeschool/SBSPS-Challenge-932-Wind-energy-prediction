# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:29:14 2020

@author: Janakiram J
"""


__author__ = 'kai'

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('first.html')

@app.route('/hello', methods=['POST'])
def hello():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    return 'Hello %s %s have fun learning python <br/> <a href="/">Back Home</a>' % (first_name, last_name)

if __name__ == '__main__':
    app.run(debug=False)