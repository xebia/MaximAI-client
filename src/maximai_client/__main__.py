import logging
from pathlib import Path
import tempfile
import re
from typing import Annotated

import dotenv
import httpx
import pvcheetah
import pvporcupine
import pvorca
from pvrecorder import PvRecorder
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.playback import play
import typer
import num2words
# import wave
import numpy as np

API_ERROR_RESPONSE = "Sorry, I didn't understand. Can you repeat what you said?"
END_PHRASES = {"by max.", "bye max.", "goodbye max."}


class Prompt(BaseModel):
    text: str
    user_id: str


class Result(BaseModel):
    input_message: str
    output_message: str
    user_id: str


def main(
    *,
    user_id: Annotated[str, typer.Option()],
    access_key: Annotated[str, typer.Option(envvar="PV_ACCESS_KEY")],
    wakeword_model_path: Path = "models/wakeword.ppn",
    endpoint_duration_sec: float = 1.0,
    disable_automatic_punctuation: bool = False,
    show_audio_devices: bool = False,
    audio_device_index: int = -1,
):
    if show_audio_devices:
        for index, name in enumerate(PvRecorder.get_available_devices()):
            print("Device #%d: %s" % (index, name))
        return

    logging.basicConfig(format="[%(asctime)s] - %(levelname)-8s - %(message)s")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    porcupine = pvporcupine.create(
        access_key=access_key, keyword_paths=[wakeword_model_path]
    )

    cheetah = pvcheetah.create(
        access_key=access_key,
        # library_path=args.library_path,
        # model_path=args.model_path,
        endpoint_duration_sec=endpoint_duration_sec,
        enable_automatic_punctuation=not disable_automatic_punctuation,
    )

    # orca = pvorca.create(access_key=access_key)

    try:
        recorder = PvRecorder(
            frame_length=cheetah.frame_length, device_index=audio_device_index
        )

        logging.info("Listening... (press Ctrl+C to stop)")

        # listen_conversation(recorder, cheetah)

        while True:
            listen_wake(recorder, porcupine, logger=logger)

            while True:
                query = listen_input(recorder, cheetah, logger=logger)

                if query.lower() in END_PHRASES:
                    # TODO: Add name!
                    # say_response(orca=orca, response="Goodbye!")
                    break

                response = generate_response_mp3(query, user_id=user_id, logger=logger)
                song = AudioSegment.from_mp3(str(response))
                play(song)
                # say_response(orca=orca, response=response, logger=logger)
    except KeyboardInterrupt:
        pass
    except (
        pvcheetah.CheetahActivationLimitError,
        pvporcupine.PorcupineActivationLimitError,
    ):
        logging.error("AccessKey has reached its processing limit.")
    finally:
        porcupine.delete()
        cheetah.delete()
        # orca.delete()


def listen_wake(recorder, porcupine, logger):
    logger.info("Listening for wakeword")
    recorder.start()

    try:
        keyword_index = -1
        while keyword_index == -1:
            keyword_index = porcupine.process(recorder.read())
    finally:
        recorder.stop()

    logger.info("Detected wakeword!")


def listen_input(recorder, cheetah, logger) -> str:
    logger.info("Listening for input")
    recorder.start()

    try:
        is_endpoint = False
        transcript = ""
        while not is_endpoint:
            partial_transcript, is_endpoint = cheetah.process(recorder.read())
            transcript += partial_transcript
            # print(partial_transcript, end="", flush=True)
        transcript += cheetah.flush()

        logger.debug(f"Received input: {transcript}")
    finally:
        recorder.stop()

    logger.info("Finished input")

    return transcript


# def listen_input_wav(recorder, cheetah, logger):
#     logger.info("Listening for input")
#     recorder.start()
#     audiodata = []
#
#     try:
#         is_endpoint = False
#
#         while not is_endpoint:
#             audiodata.append(recorder.read())
#
#     finally:
#         recorder.stop()
#         audio_data_np = np.concatenate(audiodata)
#         with wave.open('keyword_detected.wav', 'w') as wav_file:
#             wav_file.setnchannels(1)
#             wav_file.setsampwidth(recorder.sample_width)
#             wav_file.setframerate(recorder.sample_rate)
#             wav_file.writeframes(audio_data_np.tobytes())
#
#         logger.info("Finished input")
#
#         return 'keyword_detected.wav'

def generate_response(query: str, user_id: str, logger) -> str:
    prompt = Prompt(text=query, user_id=user_id)

    logger.info("Sending prompt")
    logging.debug(f"Prompt value: {prompt}")

    try:
        response = httpx.post(
            url="https://maximai-cnc2gks64q-ez.a.run.app/chat",
            json=prompt.model_dump(),
            timeout=10.0,
        )
        response.raise_for_status()

        result = Result(**response.json())
        logger.debug(f"Received response: {result}")

        return result.output_message
    except Exception as exc:
        logger.error(exc)
        return API_ERROR_RESPONSE


def generate_response_mp3(query: str, user_id: str, logger, output_wav_file: str = "response.mp3") -> str:
    prompt = Prompt(text=query, user_id=user_id)

    logger.info("Sending prompt")
    logging.debug(f"Prompt value: {prompt}")

    try:
        response = httpx.post(
            url="https://maximai-v2-cnc2gks64q-ez.a.run.app/text_audio",
            json=prompt.model_dump(),
            timeout=10.0,
        )
        response.raise_for_status()

        # Write the response content to a .wav file
        with open(output_wav_file, 'wb') as f:
            f.write(response.content)

        return output_wav_file
    except Exception as exc:
        logger.error(exc)
        return API_ERROR_RESPONSE

def say_response(orca, response, logger, file_name="speech.wav"):
    logger.info("Saying response")
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / file_name

        response_sanitized = sanitize(response)
        logger.debug(f"Response text: {response_sanitized}")
        orca.synthesize_to_file(
            response_sanitized, output_path=str(output_path), speech_rate=1.0
        )

        song = AudioSegment.from_wav(str(output_path))
        play(song)


def sanitize(text: str) -> str:
    sanitized= re.sub(
        "[^a-zA-Z0-9\.\,\-\s'?!:]",
        "",
        text.replace("\n", " ").replace("\r", " ").replace("’", "'"),
    )
    sanitized = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), sanitized)
    return sanitized


if __name__ == "__main__":
    dotenv.load_dotenv()
    typer.run(main)
