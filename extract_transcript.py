import os
import argparse
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import pdb

def extract_audio(video_filepath, output_filepath):
    video_clip = VideoFileClip(video_filepath)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_filepath, codec='libmp3lame')
    video_clip.close()
    audio_clip.close()

def extract_transcript(audio_filepath, output_filepath):
    model_id = "openai/whisper-large-v3"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)

    # Initialize the pipeline for automatic speech recognition
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model=model_id,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Process the audio file directly
    result = asr_pipeline(audio_filepath)

    final_text = ""
    for chunk in result['chunks']:
        t1 = format_time(chunk['timestamp'][0])
        t2 = format_time(chunk['timestamp'][1])
        text = chunk['text'].strip()
        final_text += f"{t1} - {t2}\n{text}\n\n"
    final_text = final_text.strip()

    with open(output_filepath, 'w') as file:
        file.write(final_text)

def format_time(seconds):
    # Calculate hours, minutes and remaining seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    # Format the time string
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:05.2f}"

def main():
    parser = argparse.ArgumentParser(description='Extract audio from video and transcribe.')
    parser.add_argument('video_filepath', type=str, help='Path to the video file.')
    parser.add_argument('-o', '--output', type=str, default='output.srt', help='Output file path (default: output.srt)')

    args = parser.parse_args()

    filepath, _ = os.path.splitext(args.output)
    audio_filepath = filepath + ".mp3"

    extract_audio(args.video_filepath, audio_filepath)
    print(f"Audio extracted and saved to {audio_filepath}")

    extract_transcript(audio_filepath, args.output)
    print(f"Transcript extracted and saved to {args.output}")

if __name__ == '__main__':
    main()
