#
#    Copyright 2018-2023 Picovoice Inc.
#
#    You may not use this file except in compliance with the license. A copy of the license is located in the "LICENSE"
#    file accompanying this source.
#
#    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
#    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
#    specific language governing permissions and limitations under the License.
#

import logging
from pathlib import Path
import tempfile
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

    orca = pvorca.create(access_key=access_key)

    try:
        recorder = PvRecorder(
            frame_length=cheetah.frame_length, device_index=audio_device_index
        )

        logging.info("Listening... (press Ctrl+C to stop)")

        # listen_conversation(recorder, cheetah)

        try:
            while True:
                listen_wake(recorder, porcupine, logger=logger)

                while True:
                    query = listen_query(recorder, cheetah, logger=logger)

                    if query.lower() in END_PHRASES:
                        # TODO: Add name!
                        say_response(orca=orca, response="Goodbye!")
                        break

                    response = generate_response(query, user_id=user_id, logger=logger)
                    say_response(orca=orca, response=response)
        finally:
            print()
            recorder.stop()
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
        orca.delete()


def listen_wake(recorder, porcupine, logger):
    logger.info("Listing for wakeword")
    recorder.start()
    keyword_index = -1
    while keyword_index == -1:
        keyword_index = porcupine.process(recorder.read())
    recorder.stop()
    logger.info("Detected wakeword!")


def listen_query(recorder, cheetah, logger) -> str:
    logger.info("Listening for query")
    recorder.start()

    is_endpoint = False
    transcript = ""
    while not is_endpoint:
        partial_transcript, is_endpoint = cheetah.process(recorder.read())
        transcript += partial_transcript
        # print(partial_transcript, end="", flush=True)
    transcript += cheetah.flush()

    logger.debug(f"Received query: {transcript}")

    recorder.stop()
    logger.info("Finished query")

    return transcript


def generate_response(query: str, user_id:str, logger) -> str:
    prompt = Prompt(
        text=query,
        user_id=user_id
    )

    logger.info("Sending prompt")
    logging.debug(f"Prompt value: {prompt}")

    try:
        response = httpx.post(
            url="https://maximai-cnc2gks64q-ez.a.run.app/chat",
            json=prompt.model_dump()
        )
        response.raise_for_status()

        result = Result(**response.json())
        logger.debug(f"Received response: {result}")

        return result.output_message
    except Exception as exc:
        logger.error(exc)
        return API_ERROR_RESPONSE


def say_response(orca, response, file_name="speech.wav"):
    logging.info("Generating response")
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / file_name
        orca.synthesize_to_file(response, output_path=str(output_path), speech_rate=1.0)

        song = AudioSegment.from_wav(str(output_path))
        play(song)


if __name__ == "__main__":
    dotenv.load_dotenv()
    typer.run(main)