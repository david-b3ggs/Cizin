file = open("quakes.csv", "r")
dest = open("quakeList.csv", "w")

line = file.readline()
dest.write(line + '\n')
seperator = ','

line = file.readline()
while line != "":
    array = line.split()
    nArray = array[0:6]
    nLine = seperator.join(nArray)
    dest.write(nLine + "\n")
    line = file.readline()

file.close()
dest.close()
