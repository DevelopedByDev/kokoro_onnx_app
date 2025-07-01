"""
pip install -U kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
python examples/save.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def split_text(text, max_sentences=2):
    sentences = text.split('. ')
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = '. '.join(sentences[i:i+max_sentences])
        if not chunk.endswith('.'):
            chunk += '.'
        chunks.append(chunk)
    return chunks

def generate_audio(kokoro, chunk, filename):
    """Generate audio for a chunk and save to file"""
    print(f"🔄 Generating: {chunk[:50]}...")
    samples, sample_rate = kokoro.create(chunk, voice="af_sarah", speed=1.2)
    sf.write(filename, samples, sample_rate, format="WAV")
    print(f"✅ Generated: {filename}")
    return filename

def play_audio_async(filename):
    """Play audio file asynchronously using afplay"""
    print(f"🔊 Playing: {filename}")
    return subprocess.Popen(['afplay', filename])

kokoro = Kokoro("kokoro-v1.0.onnx", "voices.json")

with open("test.txt", "r") as file:
    text = file.read()

chunks = split_text(text)
print(f"📝 Split text into {len(chunks)} chunks")

# Use ThreadPoolExecutor with 4 workers for parallel generation
with ThreadPoolExecutor(max_workers=4) as executor:
    print("🚀 Starting parallel generation of all chunks...")
    
    # Submit all chunks for generation in parallel
    futures = []
    for i, chunk in enumerate(chunks):
        filename = f"audio_{i}.wav"
        future = executor.submit(generate_audio, kokoro, chunk, filename)
        futures.append((future, filename, i))
    
    # Wait for all generations to complete
    generated_files = []
    for future, filename, index in futures:
        result_filename = future.result()
        generated_files.append((index, result_filename))
        print(f"✅ Chunk {index + 1}/{len(chunks)} generation completed")
    
    # Sort by index to ensure correct playback order
    generated_files.sort(key=lambda x: x[0])
    
    print("🎵 Starting sequential playback...")
    
    # Play all files sequentially
    for i, (index, filename) in enumerate(generated_files):
        print(f"🔊 Playing chunk {i + 1}/{len(chunks)}: {filename}")
        audio_process = play_audio_async(filename)
        
        # Wait for current audio to finish before starting next one
        audio_process.wait()
        print(f"✅ Finished playing chunk {i + 1}")

print("🎉 All audio chunks have been generated and played!")    
