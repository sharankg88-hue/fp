from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tempfile
import os
import numpy as np
import soundfile as sf
from PIL import Image
import cv2
from moviepy.editor import VideoFileClip  # (currently unused but kept if you want to extend)
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import serialization, hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# ---------------------------------------------
# CRYPTO FUNCTIONS
# ---------------------------------------------
def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    public_key = private_key.public_key()

    priv = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    pub = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return priv.decode(), pub.decode()


def encrypt_hybrid(message: str, public_pem: str):
    public_key = serialization.load_pem_public_key(
        public_pem.encode(), backend=default_backend()
    )

    aes_key = os.urandom(32)
    iv = os.urandom(16)

    padder = padding.PKCS7(128).padder()
    padded = padder.update(message.encode()) + padder.finalize()

    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    encrypted_aes_key = public_key.encrypt(
        aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # packed layout: [ RSA(aes_key)=256 bytes | iv=16 bytes | ciphertext ]
    return encrypted_aes_key + iv + ciphertext


def decrypt_hybrid(packed: bytes, private_pem: str):
    private_key = serialization.load_pem_private_key(
        private_pem.encode(), password=None, backend=default_backend()
    )

    encrypted_aes_key = packed[:256]
    iv = packed[256:256 + 16]
    ciphertext = packed[256 + 16:]

    aes_key = private_key.decrypt(
        encrypted_aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode()

# ---------------------------------------------
# STEGANOGRAPHY HELPERS (COMMON)
# ---------------------------------------------
def pack_payload(payload: bytes):
    # 4-byte big-endian length prefix + payload
    return len(payload).to_bytes(4, "big") + payload

# ---------------------------------------------
# IMAGE STEGO
# ---------------------------------------------
def embed_bytes_in_image_file(in_path, payload, out_path):
    img = Image.open(in_path).convert("RGB")
    arr = np.array(img)
    flat = arr.reshape(-1)

    packed = pack_payload(payload)
    bits = np.array([int(b) for byte in packed for b in f"{byte:08b}"], dtype=np.uint8)

    if len(bits) > flat.size:
        raise ValueError("Image too small to hold payload")

    flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits

    stego = flat.reshape(arr.shape)
    Image.fromarray(stego).save(out_path)
    return out_path


def extract_bytes_from_image_file(in_path):
    img = Image.open(in_path).convert("RGB")
    flat = np.array(img).reshape(-1)

    header = flat[:32] & 1
    length = int("".join(str(b) for b in header), 2)

    total = 32 + length * 8
    if total > flat.size:
        raise ValueError("Corrupted or incomplete stego image")

    bits = flat[32:total] & 1

    data = [
        int("".join(str(bit) for bit in bits[i:i + 8]), 2)
        for i in range(0, len(bits), 8)
    ]

    return bytes(data)

# ---------------------------------------------
# TEXT STEGO — zero-width characters
# ---------------------------------------------
ZW0 = "\u200B"
ZW1 = "\u200C"
DELIM = "\u200D"

def embed_bytes_in_text_file(in_path, payload, out_path):
    text = open(in_path, "r", encoding="utf-8").read()
    packed = pack_payload(payload)
    bits = ''.join(f"{b:08b}" for b in packed)
    hidden = ''.join(ZW0 if b == "0" else ZW1 for b in bits)
    open(out_path, "w", encoding="utf-8").write(text + DELIM + hidden)
    return out_path


def extract_bytes_from_text_file(in_path):
    content = open(in_path, "r", encoding="utf-8").read()
    if DELIM not in content:
        raise ValueError("No hidden data found in text file")

    hidden = content.split(DELIM)[-1]
    bits = ''.join("0" if c == ZW0 else "1" for c in hidden)

    if len(bits) < 32:
        raise ValueError("Corrupted or incomplete stego text")

    length = int(bits[:32], 2)
    data_bits = bits[32:32 + length * 8]

    if len(data_bits) < length * 8:
        raise ValueError("Corrupted or incomplete stego text")

    data = bytes(int(data_bits[i:i + 8], 2) for i in range(0, len(data_bits), 8))
    return data

# ---------------------------------------------
# AUDIO STEGO (WAV; LSB on PCM samples)
# ---------------------------------------------
def embed_bytes_in_audio_file(in_path, payload, out_path):
    # Read audio as 16-bit PCM
    data, samplerate = sf.read(in_path, dtype='int16')   # data is int16 or can be cast
    arr = np.array(data, dtype=np.int16)

    # Work in int32 to avoid overflow problems, then cast back to int16
    flat = arr.reshape(-1).astype(np.int32)

    packed = pack_payload(payload)
    bits = np.array(
        [int(b) for byte in packed for b in f"{byte:08b}"],
        dtype=np.int32
    )

    if len(bits) > flat.size:
        raise ValueError("Audio file too small to hold payload")

    # Clear LSB and set it to payload bit.
    # Use ~1 instead of 0xFFFE so it matches the signed type.
    flat[:len(bits)] = (flat[:len(bits)] & ~1) | bits

    # Clip back into int16 range and reshape
    flat = np.clip(flat, -32768, 32767).astype(np.int16)
    stego = flat.reshape(arr.shape)

    # Write back as 16-bit PCM WAV
    sf.write(out_path, stego, samplerate, subtype='PCM_16')
    return out_path
# ---------------------------------------------
# VIDEO STEGO (LSB on frames using OpenCV)
# ---------------------------------------------
def extract_bytes_from_video_file(in_path):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")

    bits = []

    # First collect at least 32 bits for length
    while len(bits) < 32:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Video too short or not a valid stego video")

        flat = frame.reshape(-1)
        take = min(32 - len(bits), flat.size)
        bits.extend(list(flat[:take] & 1))

    length = int("".join(str(b) for b in bits[:32]), 2)
    total_bits_needed = 32 + length * 8

    while len(bits) < total_bits_needed:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Corrupted or incomplete stego video")

        flat = frame.reshape(-1)
        remaining = total_bits_needed - len(bits)
        take = min(remaining, flat.size)
        bits.extend(list(flat[:take] & 1))

    cap.release()

    data_bits = bits[32:total_bits_needed]
    data_bytes = [
        int("".join(str(bit) for bit in data_bits[i:i + 8]), 2)
        for i in range(0, len(data_bits), 8)
    ]

    return bytes(data_bytes)
import subprocess

def merge_video_and_audio(stego_video_path, original_video_path, final_output_path):
    """
    Uses FFmpeg to merge audio from original video into the stego video.
    Both must have the same number of frames & duration roughly.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", stego_video_path,
        "-i", original_video_path,
        "-c:v", "copy",        # copy video stream (stego frames)
        "-c:a", "aac",         # re-encode audio to AAC
        "-map", "0:v:0",       # take video from stego
        "-map", "1:a:0?",      # take audio from original (if exists)
        "-shortest",
        final_output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        raise RuntimeError(f"FFmpeg merge failed: {e}")

    return final_output_path

def embed_bytes_in_video_file(in_path, payload, out_path):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError("Cannot open input video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback

    if frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video properties")

    packed = pack_payload(payload)
    bits = np.array(
        [int(b) for byte in packed for b in f"{byte:08b}"],
        dtype=np.uint8,
    )

    total_capacity = frame_count * width * height * 3  # 3 channels (BGR)
    if len(bits) > total_capacity:
        cap.release()
        raise ValueError("Video too small to hold payload")

    # IMPORTANT: write to AVI with MJPG (less destructive than mp4v)
    # Still not truly lossless, but much better for LSB than H.264/mp4v
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), True)

    bit_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if bit_idx < len(bits):
            flat = frame.reshape(-1)
            remaining = len(bits) - bit_idx
            chunk_size = min(remaining, flat.size)

            flat[:chunk_size] = (flat[:chunk_size] & 0xFE) | bits[bit_idx:bit_idx + chunk_size]
            bit_idx += chunk_size

            frame = flat.reshape(frame.shape)

        out.write(frame)

    cap.release()
    out.release()

    if bit_idx < len(bits):
        raise ValueError("Not all bits were embedded into the video (unexpected)")

    return out_path


# ---------------------------------------------
# FLASK API
# ---------------------------------------------
app = Flask(__name__)
CORS(app)

@app.get("/generate_keys")
def api_generate_keys():
    priv, pub = generate_rsa_keys()
    return jsonify({"private_key": priv, "public_key": pub})


@app.post("/encrypt_embed")
def api_encrypt_embed():
    try:
        pubkey = request.form["public_key"]
        message = request.form["message"]
        media = request.files["media"]

        encrypted = encrypt_hybrid(message, pubkey)

        temp_in = tempfile.NamedTemporaryFile(delete=False).name
        temp_out = tempfile.NamedTemporaryFile(delete=False).name

        media.save(temp_in)
        name = media.filename.lower()

        # Decide output path + mimetype + download name based on input type
        if name.endswith((".png", ".jpg", ".jpeg")):
            outfile = temp_out + ".png"
            mimetype = "image/png"
            download_name = "stego_image.png"
            embed_bytes_in_image_file(temp_in, encrypted, outfile)

        elif name.endswith(".txt"):
            outfile = temp_out + ".txt"
            mimetype = "text/plain"
            download_name = "stego_text.txt"
            embed_bytes_in_text_file(temp_in, encrypted, outfile)

        elif name.endswith(".wav"):
            outfile = temp_out + ".wav"
            mimetype = "audio/wav"
            download_name = "stego_audio.wav"
            embed_bytes_in_audio_file(temp_in, encrypted, outfile)

        elif name.endswith((".mp4", ".avi", ".mov", ".mkv")):
            silent_stego = temp_out + "_silent.avi"     # AVI temp for OpenCV
            final_video = temp_out + ".mp4"             # MP4 with audio merged

            embed_bytes_in_video_file(temp_in, encrypted, silent_stego)
            merge_video_and_audio(silent_stego, temp_in, final_video)

            outfile = final_video
            mimetype = "video/mp4"
            download_name = "stego_video.mp4"




        else:
            return "Unsupported file", 400

        return send_file(
            outfile,
            mimetype=mimetype,
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        app.logger.exception("Error in /encrypt_embed")
        return f"Error in encrypt_embed: {e}", 500


@app.post("/decrypt")
def api_decrypt():
    try:
        priv = request.form["private_key"]
        media = request.files["media"]

        temp_in = tempfile.NamedTemporaryFile(delete=False).name
        media.save(temp_in)
        name = media.filename.lower()

        if name.endswith((".png", ".jpg", ".jpeg")):
            payload = extract_bytes_from_image_file(temp_in)
        elif name.endswith(".txt"):
            payload = extract_bytes_from_text_file(temp_in)
        elif name.endswith(".wav"):
            payload = extract_bytes_from_audio_file(temp_in)
        elif name.endswith((".mp4", ".avi", ".mov", ".mkv")):
            payload = extract_bytes_from_video_file(temp_in)
        else:
            return "Unsupported file", 400

        message = decrypt_hybrid(payload, priv)
        return jsonify({"message": message})

    except Exception as e:
        app.logger.exception("Error in /decrypt")
        return f"Error in decrypt: {e}", 500


@app.get("/")
def index():
    return "CryptAVIT Backend Running"

# ---------------------------------------------
# DEPLOY
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

