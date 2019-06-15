import json


class FloydLogger(object):
    def __init__(self):
        pass

    def log(self, metric, value, step=None):
        dic = {"metric": metric, "value": value}
        if step is not None:
            dic['step'] = step

        print(json.dumps(dic))
