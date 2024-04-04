from flask import Flask, request, redirect, jsonify
from flask_restful import Api, Resource
import json
import numpy as np
import pandas as pd
import win_estimator
from flask.json.provider import DefaultJSONProvider


class NumpyArrayEncoder(DefaultJSONProvider):
    def default(self, obj):
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                if obj.dtype == np.int32:
                    return int(obj)  # Convert int32 to Python int
                return obj.item()  # For other NumPy integers
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                raise TypeError(" {}".format(obj))
        except AttributeError as e:
            return "Attribute error: {}".format(e)
        except TypeError as e:
            return "Type error: {}".format(e)


class CustomizedFlask(Flask):
    json_provider_class = NumpyArrayEncoder


app = CustomizedFlask(__name__)
# app = Flask(__name__)
api = Api(app)


class GetSampleData(Resource):

    def get(self):
        play = request.args.get('play')
        size = request.args.get('size')
        if play is not None and size is not None:
            response_data = win_estimator.sample_generator(play=int(play), size=int(size))
        else:
            response_data = "Incorrect data"
        return response_data


class GameSimulation(Resource):
    """"
    sample request:
    {"data": {"user_play": [[61, 70, 92, 20, 15, 98, 13, 77, 74, 31],[90, 55, 15, 25, 11, 67, 73, 55, 56,  4],
    [48, 66, 31, 97, 67, 83, 50, 17, 49, 76], [35, 76,  0,  3, 37, 91, 47, 72, 45, 38],
    [81, 37, 38, 25, 31,  8, 92,  8, 18, 68],[78, 16, 55, 14, 92, 47, 66, 95, 60, 78],
    [97, 53, 19, 92, 14, 62,  9, 27, 67, 64],[58, 44, 78, 58, 48, 24, 97, 28, 33, 57],
    [70, 75, 90, 16, 16, 42,  9, 68,  3, 36]], "user_Id": ["9eNar", "dpCmv", "XBEEH", "Ibaug", "7Gzes",
    "iUHi4", "0Om7v", "MFJjp", "BxdPN"], "stake": [84, 62, 62, 89, 87, 98, 85, 74, 70]},
    "result": [70, 75, 90, 16, 16, 42,  9, 68,  3, 36],
    "game": ["exact_1","any_2"],
    "per_profit": 50}
    """

    def post(self):
        try:
            data_param = request.json
            data_param["result"] = np.array(data_param["result"]).tolist()

            response_data = win_estimator.multiGameChecker(test_data=data_param["data"],
                                                           result_2=data_param["result"],
                                                           game=data_param["game"],
                                                           per_profit=data_param["per_profit"])
        except KeyError as e:
            return "KeyError: {}".format(e)
        except TypeError as e:
            return "TypeError: {}".format(e)
        except AttributeError as e:
            return "AttributeError: {}".format(e)

        return jsonify(response_data)




class WinNumber(Resource):
    """
    This class contains a post endpoint that generates the winning number based request.

    sample request:
    {"user_play": [[61, 70, 92, 20, 15, 98, 13, 77, 74, 31],[90, 55, 15, 25, 11, 67, 73, 55, 56,  4],
    [48, 66, 31, 97, 67, 83, 50, 17, 49, 76], [35, 76,  0,  3, 37, 91, 47, 72, 45, 38],
    [81, 37, 38, 25, 31,  8, 92,  8, 18, 68],[78, 16, 55, 14, 92, 47, 66, 95, 60, 78],
    [97, 53, 19, 92, 14, 62,  9, 27, 67, 64],[58, 44, 78, 58, 48, 24, 97, 28, 33, 57],
    [70, 75, 90, 16, 16, 42,  9, 68,  3, 36]], "user_Id": ["9eNar", "dpCmv", "XBEEH", "Ibaug", "7Gzes",
    "iUHi4", "0Om7v", "MFJjp", "BxdPN"], "stake": [84, 62, 62, 89, 87, 98, 85, 74, 70], "game":["exact_1",
    "any_2","any_4","exact_1","any_2","any_2","exact_1","any_2","any_3"]}

    sample response:
    {
    "doped method": {
        "win details": {
            "number_of_wins": {
                "any_2": 0,
                "any_3": 0,
                "any_4": 0,
                "exact_1": 0
            },
            "win_ratio": {
                "any_2": 0.0,
                "any_3": 0.0,
                "any_4": 0.0,
                "exact_1": 0.0
            }
        },
        "win result": [
            23,
            22,
            21,
            12,
            10,
            7,
            6,
            5,
            2,
            1
        ]
    },
    "least number method": {
        "win details": {
            "number_of_wins": {
                "any_2": 3,
                "any_3": 0,
                "any_4": 0,
                "exact_1": 0
            },
            "win_ratio": {
                "any_2": 33.33333333333333,
                "any_3": 0.0,
                "any_4": 0.0,
                "exact_1": 0.0
            }
        },
        "win result": [
            0,
            45,
            50,
            53,
            56,
            57,
            60,
            61,
            62,
            44
        ]
    },
    "random from dataset method": {
        "win details": {
            "number_of_wins": {
                "any_2": 4,
                "any_3": 3,
                "any_4": 2,
                "exact_1": 1
            },
            "win_ratio": {
                "any_2": 44.44444444444444,
                "any_3": 33.33333333333333,
                "any_4": 22.22222222222222,
                "exact_1": 11.11111111111111
            }
        },
        "win result": [
            48,
            66,
            42,
            66,
            78,
            31,
            16,
            33,
            24,
            97
        ]
    }
}
    """

    def post(self):
        try:
            data_param = request.json
            doped_result = win_estimator.dopedResult(data_set=data_param)
            least_number_result = win_estimator.leastOccuring(data_set=data_param)
            random_from_dataset = win_estimator.random_from_array(data_set=data_param)

            response_data = {
                "doped method": {"win result": doped_result
                                 },
                "least number method": {"win result": least_number_result
                                        },
                "random from dataset method": {"win result": random_from_dataset
                                               },
            }
        except KeyError as e:
            return "KeyError: {}".format(e)
        except AttributeError as e:
            return "AttributeError: {}".format(e)

        return jsonify(response_data)


api.add_resource(WinNumber, '/')
api.add_resource(GetSampleData, '/GetSampleData')
api.add_resource(GameSimulation, '/GameSimulation')

if __name__ == '__main__':
    app.run(debug=True)
