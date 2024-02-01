import os
import argparse
from moviepy.editor import VideoFileClip
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_audio(video_filepath, output_filepath):
    # Convert to MP3
    video_clip = VideoFileClip(video_filepath)
    audio_clip = video_clip.audio

    filepath, _ = os.path.splitext(output_filepath)
    temp_audio_filepath = filepath + ".temp.wav"
    audio_clip.write_audiofile(temp_audio_filepath)

    # Use pydub to convert audio to mono
    sound = AudioSegment.from_wav(temp_audio_filepath)
    sound = sound.set_channels(1)
    sound.export(output_filepath, format="mp3")

    # Close the clips
    video_clip.close()
    audio_clip.close()


def extract_transcript(audio_filepath, output_filepath):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    # # Function to load an audio file and prepare it for the pipeline
    # def prepare_audio(file_path):
    #     waveform, sample_rate = torchaudio.load(file_path)
    #     return {"array": waveform.numpy(), "sampling_rate": sample_rate}

    # Function to load an audio file and prepare it for the pipeline
    def prepare_audio(file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        # Convert to mono by selecting the first channel if stereo
        if waveform.shape[0] > 1:
            waveform = waveform[0,:].unsqueeze(0)
        return {"array": waveform.numpy(), "sampling_rate": sample_rate}
        
    # Prepare your audio file
    audio_data = prepare_audio(audio_filepath)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio_data)
    with open(output_filepath, 'a') as file:
        file.write(result['text'])


def check_audio_channels(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Number of channels in the audio: {waveform.shape[0]}")
    print(f"Sample rate of the audio: {sample_rate}")
    
def main():
    parser = argparse.ArgumentParser(description='Extract audio from video.')
    parser.add_argument('video_filepath', type=str, help='Path to the video file.')
    parser.add_argument('-o', '--output', type=str, default='output.txt', help='Output file path (default: output.txt)')

    args = parser.parse_args()

    filepath, _ = os.path.splitext(args.output)
    audio_filepath = filepath + ".mp3"

    extract_audio(args.video_filepath, audio_filepath)
    print(f"Converted to mp3 and saved to {audio_filepath}")

    check_audio_channels(audio_filepath)

    extract_transcript(audio_filepath, args.output)
    print(f"Transcript extracted and saved to {args.output}")

if __name__ == '__main__':
    main()