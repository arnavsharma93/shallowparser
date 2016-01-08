from flask import Flask
from flask_restful import Resource, Api, reqparse
from predict import *
from collections import OrderedDict

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('sentence')

def serializer(X_test):
    out = []
    for x in X_test:
        for obv in x:
            cobv = {
                    #'WORD': obv['WORD'],
                    'LANG': obv['LANG'],
                     'POS': obv['POS'],
                     'CHUNK': obv['CHUNK']
            }
            out.append({obv['WORD']:cobv})

    return out



class ShallowParser(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        args = parser.parse_args()
        X_test = shallow_parser(args['sentence'])
        return serializer(X_test)

api.add_resource(ShallowParser, '/')

if __name__ == '__main__':
    app.run(debug=True)
