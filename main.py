import os
import replicate
from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Settings
os.environ["REPLICATE_API_TOKEN"] = "r8_92jQgKdTpW59OtalNSSVMumP8Xn52Iy0BqLCD"
UPLOAD_FOLDER = 'uploads'
PRESET_FOLDER = 'presets' # Yahan aap apni pasand ki BG photos rakhenge
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRESET_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process():
    image_file = request.files.get('image')
    bg_custom = request.files.get('bg_custom') # User ki apni photo
    bg_preset = request.form.get('bg_preset') # Dropdown se chuna hua
    audio_file = request.files.get('audio')
    remove_bg = request.form.get('remove_bg') == 'true'

    try:
        input_img_path = os.path.join(UPLOAD_FOLDER, "final_character.png")
        
        if remove_bg:
            print("Processing Background...")
            char_data = image_file.read()
            no_bg_img = remove(char_data)
            character = Image.open(io.BytesIO(no_bg_img)).convert("RGBA")

            # Background chunne ka logic
            if bg_custom:
                background = Image.open(bg_custom).convert("RGBA")
            elif bg_preset and bg_preset != "none":
                # Preset folder se photo uthana
                bg_path = os.path.join(PRESET_FOLDER, f"{bg_preset}.jpg")
                background = Image.open(bg_path).convert("RGBA")
            else:
                # Default agar kuch na ho toh Safed (White)
                background = Image.new("RGBA", character.size, (255, 255, 255, 255))

            background = background.resize(character.size)
            background.paste(character, (0, 0), character)
            background.save(input_img_path)
        else:
            image_file.save(input_img_path)

        # AI Animation
        audio_path = os.path.join(UPLOAD_FOLDER, "voice.mp3")
        if audio_file:
            audio_file.save(audio_path)
        else:
            audio_path = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

        output = replicate.run(
            "cjwbw/sadtalker:3aa3fda9cd158a5654de64635da3c01accc6a627758e11f09e2ad3b7b32b2707",
            input={
                "source_image": open(input_img_path, "rb"),
                "driven_audio": open(audio_path, "rb") if audio_file else audio_path,
                "preprocess": "full",
                "enhancer": "gfpgan"
            }
        )

        return jsonify({"status": "Success", "video_url": output})

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)