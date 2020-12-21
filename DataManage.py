import time
import json

# fetch data array every 1.024 seconds, add it to the buffer

class DataManager(object):

    def collect(self, buffer):
        testFile = open("appTest.json", "r")


        for obj in testFile:
            yArr, zArr, xArr = [], [], []
            lineData = json.loads(obj)
            xArr.append(lineData["x"])
            yArr.append(lineData["y"])
            zArr.append(lineData["z"])

            data = [xArr, yArr, zArr]

            buffer.write(data)
            time.sleep(1.024)