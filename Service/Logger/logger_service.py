import datetime


class LoggerMethods():
    def __init__(self, path_log:str):
        self.logger_path = path_log

    def write(self, message):
        with open(self.logger_path + datetime.datetime.today().strftime('%Y-%m-%d') + '.log', "a") as file:
            file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")