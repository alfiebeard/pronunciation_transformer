from data.preprocessing.tokeniser import tokenise
from pydub import AudioSegment, effects
import time


def join_list_ipa_audio(ipa_list, ipa_audio_file_dir="data/datasets/processed_data/trimmed_ipa_audio"):
    for i in range(len(ipa_list)):
        if i == 0:
            ipa_sound = join_ipa_audio(ipa_list[i], ipa_audio_file_dir=ipa_audio_file_dir)
        else:
            updated_sound = AudioSegment.silent(duration=500) + join_ipa_audio(ipa_list[i], ipa_audio_file_dir=ipa_audio_file_dir)
            ipa_sound += updated_sound
    
    return ipa_sound


def join_ipa_audio(ipa, ipa_audio_file_dir="data/datasets/processed_data/trimmed_ipa_audio", speed_up=4):
    ipa_tokenised = tokenise(ipa)

    ipa_sound = None
    for i in range(len(ipa_tokenised)):
        ipa_sound_file = AudioSegment.from_mp3(ipa_audio_file_dir + "/" + ipa_tokenised[i] + ".mp3")

        # If it's the first item in sequence, initialise the full sound, otherwise add on
        if i == 0:
            ipa_sound = ipa_sound_file
        else:
            ipa_sound = ipa_sound.append(ipa_sound_file, crossfade=130)
    
    ipa_sound = ipa_sound.speedup(speed_up, crossfade=130)

    return ipa_sound


def save_audio(ipa_sound, word_list=None, saved_audio_dir="speech_model/saved_audio_predictions"):
    if word_list:
        audio_filename = "_".join(word_list)[:20]
    else:
        audio_filename = time.strftime("%Y%m%d_%H%M%S")
    # Save file
    if ipa_sound:
        ipa_sound.export(saved_audio_dir + "/" + audio_filename + '.mp3', format="mp3")


def trim_ipa_audio(phonemes=['b', 'd', 'g', 'm', 'n', 'p', 't', 'ɹ', 'k', 'tʃ', 'f', 'ʤ', 'l', 's', 'v', 'z', 'h', 'ŋ', 'ʃ', 'ʒ', 'θ', 'ð', 'w', 'j', 'æ', 'e', 'ɪ', 'ɒ', 'ʌ', 'ʊ', 'ə', 'ɑː', 'ɜː', 'ɔː', 'uː', 'iː', 'eɪ', 'aɪ', 'əʊ', 'aʊ', 'ɔɪ', 'eə', 'ɪə', 'ʊə'], ipa_audio_file_dir="data/datasets/ipa_audio", saved_audio_dir="data/datasets/trimmed_ipa_audio"):
    for phoneme in phonemes:
        ipa_sound_file = AudioSegment.from_mp3(ipa_audio_file_dir + "/" + phoneme + ".mp3")
        trimmed_ipa_sound_file = trim_silence(ipa_sound_file)
        trimmed_ipa_sound_file.export(saved_audio_dir + "/" + phoneme + ".mp3", format="mp3")


def trim_silence(sound, silence_threshold=-50.0, chunk_size=1):
    trim_start_ms = 0
    trim_end_ms = len(sound)

    assert chunk_size > 0 # to avoid infinite loop
   
    # Silence at start
    while sound[trim_start_ms:trim_start_ms+chunk_size].dBFS < silence_threshold and trim_start_ms < len(sound):
        trim_start_ms += chunk_size

    # Silence at end
    while sound[trim_end_ms-chunk_size:trim_end_ms].dBFS < silence_threshold and trim_end_ms - chunk_size > 0:
        trim_end_ms -= chunk_size
  
    return sound[trim_start_ms:trim_end_ms]


def test():
    phonemes = ['b', 'd', 'g', 'm', 'n', 'p', 't', 'ɹ', 'k', 'tʃ', 'f', 'ʤ', 'l', 's', 'v', 'z', 'h', 'ŋ', 'ʃ', 'ʒ', 'θ', 'ð', 'w', 'j', 'æ', 'e', 'ɪ', 'ɒ', 'ʌ', 'ʊ', 'ə', 'ɑː', 'ɜː', 'ɔː', 'uː', 'iː', 'eɪ', 'aɪ', 'əʊ', 'aʊ', 'ɔɪ', 'eə', 'ɪə', 'ʊə']
    for i in range(len(phonemes)):
        filename = "data/datasets/trimmed_ipa_audio" + "/" + phonemes[i] + ".mp3"
        ipa_sound_file = AudioSegment.from_mp3(filename)

        # If it's the first item in sequence, initialise the full sound, otherwise add on
        if i == 0:
            ipa_sound = ipa_sound_file
        else:
            ipa_sound += ipa_sound_file
    
    ipa_sound.export("speech_model/saved_audio_predictions" + "/" + "test.mp3", format="mp3")


if __name__ == "__main__":
    # trim_ipa_audio()
    aud = join_ipa_audio("hɛləʊ")
    save_audio(aud)
