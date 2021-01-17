import pandas as pd
import numpy as np
import sqlite3

from flask import Flask, request, jsonify, redirect, g
from flask_restful import Resource, Api
from webargs import fields, validate
from webargs.flaskparser import use_args, use_kwargs, parser, abort
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)
clf = joblib.load('pipeline.pkl')

FILEDS = ['Year', 'Mileage', 'City', 'State', 'Vin', 'Make', 'Model']


data_args = {
    "Price": fields.Str(required=False),
    "Year": fields.Str(required=True),
    "Mileage": fields.Str(required=True),
    "City": fields.Str(required=True),
    "State": fields.Str(required=True),
    "Vin": fields.Str(required=True),
    "Make": fields.Str(required=True),
    "Model": fields.Str(required=True),
}

DATABASE = 'data.db'
TABLE = 'car_data'


def make_dicts(cursor, row):
    return dict((cursor.description[idx][0], value)
                for idx, value in enumerate(row))


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = make_dicts
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/fetchData', methods=['GET'])
def fetach_data():
    size = request.args.get('size', '')
    db = get_db()
    sql = 'SELECT {} FROM {}'.format('*', TABLE)
    rv = db.execute(sql).fetchall()
    return jsonify(rv)


class HelloWorld(Resource):
    def get(self):
        return {'msg': 'please read the document to use the APIs.'}


class Add(Resource):
    @use_kwargs(data_args)
    def get(self, Price, Year, Mileage, City, State, Vin, Make, Model):
        content = ','.join(
            [Price, Year, Mileage, City, State, Vin, Make, Model])
        with open('add.csv', 'a') as f:
            f.write(content + '\n')
        db = get_db()
        sql = "INSERT INTO {} VALUES ({})".format(TABLE, content)
        return {'msg': 'success'}


class Predict(Resource):
    @use_kwargs(data_args)
    def get(self, Price, Year, Mileage, City, State, Vin, Make, Model):
        x = [Year, Mileage, City, State, Vin, Make, Model]
        x = pd.DataFrame([x], columns=FILEDS)
        price = clf.predict(x)
        return {'Price': float(price[0])}

    @use_kwargs(data_args)
    def post(self, Price, Year, Mileage, City, State, Vin, Make, Model):
        x = [Year, Mileage, City, State, Vin, Make, Model]
        x = pd.DataFrame([x], columns=FILEDS)
        price = clf.predict(x)
        return {'Price': float(price[0])}


@parser.error_handler
def handle_request_parsing_error(err, req, schema):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(422, errors=err.messages)


api.add_resource(HelloWorld, '/')
api.add_resource(Add, '/add')
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
