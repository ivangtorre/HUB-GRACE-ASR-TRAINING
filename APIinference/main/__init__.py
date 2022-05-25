import os
import sys
import time
from traceback import print_exc
from flask import Flask, request, make_response, jsonify
from .parsing_checkers.request_checker import JSONError
from .wav2vec2.inference_server import inference_wav2vec2
from .helpers import read_audio, convert2b64, format_result, convert_audio_before_decode
from threading import Thread
from .management.broker_management import Broker
from .management.grace_protocol_data_formats import CompletedRestBrokerOutput, ProtocolOperation


class API(Flask):
    def __init__(self):
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None

        super().__init__(__name__)
        self.use_end = bool(os.environ.get('USE_END', ""))
        self.asr_en = inference_wav2vec2("/workspace/models/en/wav2vec2model", "/workspace/models/en/LM/LM.bin")
        self.asr_es = inference_wav2vec2("/workspace/models/es/wav2vec2model", "/workspace/models/es/LM/LM.bin")
        self.asr_fr = inference_wav2vec2("/workspace/models/fr/wav2vec2model", "/workspace/models/fr/LM/LM.bin")
        self.asr_pt = inference_wav2vec2("/workspace/models/pt/wav2vec2model", "/workspace/models/pt/LM/LM.bin")

        self.generate_routes()

    def asr(self, language, file_wav, keywords):
        if language == "es":
            transcript_df = self.asr_es.transcribe(file_wav)
        elif language == "en":
            transcript_df = self.asr_en.transcribe(file_wav)
        elif language == "fr":
            transcript_df = self.asr_fr.transcribe(file_wav)
        elif language == "pt":
            transcript_df = self.asr_pt.transcribe(file_wav)
        os.system("rm " + file_wav)
        return format_result(transcript_df, keywords)

    def orchest_asr(self, language, file_wav, keywords, request_data):
        start_time = time.time()
        asr_result = self.asr(language, file_wav, keywords)
        end_time = time.time()

        broker = Broker(server=request_data["broker_info"]["broker_url"],
                        user=request_data["broker_info"]["broker_user"],
                        password=request_data["broker_info"]["broker_password"])

        rest_broker_output = CompletedRestBrokerOutput(operation=ProtocolOperation.PROCESS_COMPLETED,
                                                       description=f'Process completed in {end_time - start_time:.2f} seconds',
                                                       context_data=request_data["context_data"],
                                                       input_data=request_data["input_data"],
                                                       output_data={
                                                           "transcription": " ".join(asr_result["words"].tolist())})
        print(rest_broker_output.json(), flush=True)
        broker.send_message(message=rest_broker_output.json(), topic=request_data["broker_info"]["topic"])

    def generate_routes(self):
        @self.route('/', methods=['GET'])
        def init():
            return jsonify({'message': 'ASR is on'})

        # DECODE FROM FRONTEND
        @self.route('/json', methods=['POST'])
        @self.route('/json/', methods=['POST'])
        @self.route("/upload", methods=['POST'])
        def decode_audiofile_frontend():
            try:
                request.get_json()

            except JSONError as e:
                return make_response(jsonify({'message': str(e), 'result': None, }), 400)

            except Exception as e:
                print_exc()
                return make_response(jsonify({'message': str(e), 'result': None, }), 500)

            finally:
                language, keywords, wav_name = convert_audio_before_decode(request)

                asr_result = self.asr(language, wav_name, keywords)
                if len(keywords) == 0:
                    return jsonify({'operation': "PROCESS_COMPLETED",
                                    'transcription': " ".join(asr_result["words"].tolist())
                                    })

                if len(keywords) != 0:
                    return jsonify({'operation': "PROCESS_COMPLETED",
                                    'transcription': "\n".join(asr_result["words"].tolist()),
                                    'result': asr_result.to_dict(),
                                    })

        # DECODE FROM TERMINAL PROVIDING A PATH or a file
        @self.route('/terminal', methods=['POST'])
        @self.route('/terminal/', methods=['POST'])
        def decode_audiofile_terminal():
            try:
                audio_path = request.form['audio']
                language = request.form['language']
                keywords = request.form['keywords']

            except JSONError as e:
                return make_response(jsonify({'message': str(e), 'result': None, }), 400)

            except Exception as e:
                print_exc()
                return make_response(jsonify({'message': str(e), 'result': None, }), 500)

            finally:
                # To check if one file or multiple files
                wav_name, sox_code = read_audio(audio_path)

                if len(keywords) == 0:  # FRONTEND OR NOT ORCHESTRATION
                    print("ASR", flush=True)
                    asr_result = self.asr(language, wav_name, keywords)
                    message = {'operation': "PROCESS_COMPLETED",
                               'transcription': " ".join(asr_result["words"].tolist())}
                    print(message, flush=True)
                    return jsonify(message)

                if len(keywords) != 0:  # IF KEYWORD SPOTTING
                    asr_result = self.asr(language, wav_name, keywords)
                    return jsonify({'file': "temp",
                                    'message': "Input correctly parsed",
                                    'transcription': "\n".join(asr_result["words"].tolist()),
                                    'result': asr_result.to_dict(),
                                    })

        # # DECODE FROM ORCHESTRATION
        @self.route("/request_orch/", methods=['POST'])
        @self.route("/request_orch", methods=['POST'])
        def decode_wav_orchest():
            try:
                request_data = request.get_json()

            except JSONError as e:
                return make_response(jsonify({'message': str(e), 'result': None, }), 400)

            except Exception as e:
                print_exc()
                return make_response(jsonify({'message': str(e), 'result': None, }), 500)

            finally:
                language = request_data["input_data"]["language"]
                keywords = ""
                try:
                    keywords = request_data["input_data"]["keywords"]
                except:
                    pass

                wav_name, sox_code = read_audio(request_data["input_data"]["audio"])
                if sox_code != 0:
                    return make_response(jsonify({"operation": "PROCESS_ERROR",
                                                  "description": "Audio not found or can not be accesed",
                                                  "context_data": str(
                                                      {"resource_id": request_data["context_data"]["resource_id"],
                                                       "task_id": request_data["context_data"]["workflow_id"]})})
                                         , 400)

                Thread(target=self.orchest_asr, args=(language, wav_name, keywords, request_data)).start()
                return make_response(jsonify({'message': "Process started"}), 200)
