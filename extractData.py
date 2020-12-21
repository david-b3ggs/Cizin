import os
import json

# This script generates a file of x, y, and z arrays coorisponding to one another sequentially

xData = []
yData = []
zData = []

# extract
count = 0
for file in os.listdir("./"):
    if "test_quake3.jsonl" == file:
        print("GOT IT")
        f = open("./" + file, "r")
        with f as f:
            for obj in f:
                object = json.loads(obj)
                xData.append(object["x"])
                yData.append(object["y"])
                zData.append(object["z"])

        f.close()
    count += 1

# Load into csv file

coors = []
output = open("./test_quake3.csv", "w")
output.write("x,y,z\n")

for i in range(0, len(xData)):
    for j in range(0, len(xData[i])):
        output.write(str(xData[i][j]) + "," + str(yData[i][j]) + "," + str(zData[i][j]) + "\n")


output.close()