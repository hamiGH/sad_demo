import os
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import random
from flask import Flask, request, jsonify
from spu import SpeechProcessingUnit


current_dir = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(current_dir, "log")
if not os.path.exists(log_path):
    os.mkdir(log_path)
LOG_FILENAME = os.path.join(log_path, 'server.log')
LOG_BACKUP_COUNT = 5
LOG_FILE_SIZE_BYTES = 50 * 1024 * 1024

VOICE_DECODE_ERROR = 10
PREDICTION_ERROR = 12
UNKNOWN_ERROR = 21
INVALID_FORMAT = 30
MISSING_ARGUMENTS = 40

errors = {
    VOICE_DECODE_ERROR: "VOICE_DECODE_ERROR",
    PREDICTION_ERROR: "PREDICTION_ERROR",
    UNKNOWN_ERROR: "UNKNOWN_ERROR",
    INVALID_FORMAT: "INVALID_FORMAT_REQ",
    MISSING_ARGUMENTS: "MISSING_ARGUMENTS_REQ"
}


def init_logger(app):
    handler = RotatingFileHandler(LOG_FILENAME, maxBytes=LOG_FILE_SIZE_BYTES, backupCount=LOG_BACKUP_COUNT)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    loggers = [app.logger]
    for logger in loggers:
        logger.addHandler(handler)


spu_core = SpeechProcessingUnit(fs=16000)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True

init_logger(app)


@app.before_request
def log_request():
    app.logger.debug("Request: %s %s", request.method, request.url)


class WebAppException(Exception):
    def __init__(self, error_code, status_code=None):
        Exception.__init__(self)
        self.status_code = 400
        self.exception = Exception
        self.error_code = error_code
        try:
            self.message = errors[self.error_code]
        except:
            self.error_code = UNKNOWN_ERROR
            self.message = errors[self.error_code]
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self):
        rv = dict()
        rv['status'] = 'failed'
        rv['code'] = self.error_code
        rv['message'] = self.message
        return rv


class ThrowsWebAppException(object):
    def __init__(self, error_code, status_code=None):
        self.error_code = error_code
        self.status_code = status_code

    def __call__(self, function):
        def return_function(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                raise WebAppException(self.error_code)

        return return_function


@app.errorhandler(WebAppException)
def handle_exception(error):
    app.logger.exception(error.message)
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@ThrowsWebAppException(error_code=PREDICTION_ERROR)
def vad_prediction(audio_file, threshold):
    vad_annotation = spu_core.apply_vad(file_path=audio_file, threshold=threshold, gpu_number=0)
    vad_out = {"vad_annotation": vad_annotation}
    return vad_out


@app.route('/sad', methods=['POST'])
def vad():
    if request.headers['Content-Type'].startswith('multipart/form-data'):
        try:
            # Remember the paramName was set to 'file', we can use that here to grab it
            file = request.files['file']
            # secure_filename makes sure the filename isn't unsafe to save
            temp_path = os.path.join(current_dir, "temp_data")
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
            save_path = os.path.join(temp_path, str(random.randint(0, 1000000)) + secure_filename(file.filename))
            # We need to append to the file, and write as bytes
            with open(save_path, 'ab') as temp_file:
                temp_file.seek(0, 2)
                file_byes = file.read()
                temp_file.write(file_byes)

            audio_file = save_path
            threshold = float(request.form["threshold"])
        except:
            raise WebAppException(error_code=MISSING_ARGUMENTS)
        response = jsonify(vad_prediction(audio_file, threshold))
        return response
    else:
        raise WebAppException(error_code=INVALID_FORMAT)


if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5005
    app.run(host=host, port=port, debug=True, use_reloader=False, threaded=False)
