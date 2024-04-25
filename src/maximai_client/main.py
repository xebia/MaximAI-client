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
import pvcheetah
import pvporcupine
import pvorca
from pvrecorder import PvRecorder
from pydub import AudioSegment
from pydub.playback import play
import typer

END_PHRASES = {
    "by max.",
    "bye max.",
    "goodbye max."
}


def main(*,
        access_key: Annotated[str, typer.Option(envvar="PV_ACCESS_KEY")],
        wakeword_model_path: Path = "models/wakeword.ppn",
        endpoint_duration_sec: float = 1.0,
        disable_automatic_punctuation: bool = False,
        show_audio_devices: bool = False,
        audio_device_index: int = -1
    ):
    if show_audio_devices:
        for index, name in enumerate(PvRecorder.get_available_devices()):
            print('Device #%d: %s' % (index, name))
        return

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[wakeword_model_path])

    cheetah = pvcheetah.create(
        access_key=access_key,
        # library_path=args.library_path,
        # model_path=args.model_path,
        endpoint_duration_sec=endpoint_duration_sec,
        enable_automatic_punctuation=not disable_automatic_punctuation
    )

    orca = pvorca.create(access_key=access_key)
    orca

    try:
        recorder = PvRecorder(frame_length=cheetah.frame_length, device_index=audio_device_index)

        logging.info('Listening... (press Ctrl+C to stop)')

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

                    response = generate_response(query)
                    say_response(orca=orca, response=response)
        finally:
            print()
            recorder.stop()
    except KeyboardInterrupt:
        pass
    except (pvcheetah.CheetahActivationLimitError, pvporcupine.PorcupineActivationLimitError):
        logging.error('AccessKey has reached its processing limit.')
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
        print(partial_transcript, end='', flush=True)
    transcript += cheetah.flush()

    recorder.stop()
    logger.info("Finished query")

    return transcript


def generate_response(query: str) -> str:
    return query


def say_response(orca, response, file_name="speech.wav"):
    logging.info('Generating response')
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / file_name
        orca.synthesize_to_file(response, output_path=str(output_path), speech_rate=1.0)

        song = AudioSegment.from_wav(str(output_path))
        play(song)


if __name__ == '__main__':
    dotenv.load_dotenv()
    typer.run(main)
