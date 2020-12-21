import threading
from Detection import Detector
from DataManage import DataManager

class Buffer(object):

    status = threading.Event()

    def __init__(self, val = []):
        self.lock = threading.Lock()
        self.status.clear()
        self.value = val

    def getStatus(self):
        return self.status

    def read(self, detector):
        self.lock.acquire()

        try:
            detector.setData(data=self.value)
            self.status.clear()
        finally:
            self.lock.release()


    # If the buffer is full, remove the oldest and add newest
    def write(self, data):

        if len(self.value) >= 607:
            self.lock.acquire()

            try:
                self.value.pop(0)
                self.value.append(data)
                self.status.set()
            finally:
                self.lock.release()
        else:
            self.lock.acquire()

            try:
                self.value.append(data)
                self.status.set()
            finally:
                self.lock.release()


class WarningSystem(object):
    global detected
    detector = Detector()
    manager = DataManager()


    def __init__(self):
        detected = False
        detected.lock = threading.Lock()

    def setWarning(self, observation):
        self.detected.lock.aquire()

        try:
            self.detected = observation
        finally:
            self.detected.lock.release()

    # initializes model and buffer
    def startUp(self):
        train, test = self.detector.setTrainData()
        X_train, y_train = self.detector.dataPreProcessing(train, test)
        model = self.detector.generateModel(X_train)
        self.detector.trainModel(X_train, y_train, model)
        


    # starts threads for data collection and detection thread
    # runs infinitely and starts warning thread if found
    def begin(self):
