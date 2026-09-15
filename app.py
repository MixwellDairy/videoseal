# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import tempfile
import torch
import torchaudio
import torchvision
import torchvision.transforms as T
from PIL import Image
import gradio as gr
import av

import videoseal
from videoseal.utils.display import save_video_audio_to_mp4

try:
    from audioseal import AudioSeal
    is_audioseal_installed = True
except ImportError:
    is_audioseal_installed = False

# Global model cache to avoid re-loading models on every request
MODELS_CACHE = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_video_file(video_path: str):
    """
    Read video and audio streams using PyAV or torchvision.io.read_video if available.
    Returns:
        video_tensor: [T, C, H, W] tensor
        audio_tensor: [channels, timesteps] tensor
        info: dict containing 'video_fps' and optionally 'audio_fps'
    """
    if hasattr(torchvision.io, "read_video"):
        try:
            return torchvision.io.read_video(video_path, output_format="TCHW")
        except Exception:
            pass

    container = av.open(video_path)
    video_frames = []
    v_fps = 30.0
    for stream in container.streams.video:
        if stream.average_rate:
            v_fps = float(stream.average_rate)
        break

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        video_frames.append(torch.from_numpy(img).permute(2, 0, 1))

    video_tensor = torch.stack(video_frames) if video_frames else torch.empty((0, 3, 0, 0))

    audio_frames = []
    info = {"video_fps": v_fps}

    if len(container.streams.audio) > 0:
        a_fps = 16000
        for stream in container.streams.audio:
            if stream.rate:
                a_fps = stream.rate
            break
        for frame in container.decode(audio=0):
            audio_frames.append(torch.from_numpy(frame.to_ndarray()))

        if audio_frames:
            audio_tensor = torch.cat(audio_frames, dim=1)
            info["audio_fps"] = a_fps
        else:
            audio_tensor = torch.empty((0, 0))
    else:
        audio_tensor = torch.empty((0, 0))

    return video_tensor, audio_tensor, info


def get_model(model_name: str):
    """Retrieve or load a VideoSeal model by name."""
    if model_name not in MODELS_CACHE:
        model = videoseal.load(model_name)
        model.eval()
        model.to(DEVICE)
        MODELS_CACHE[model_name] = model
    return MODELS_CACHE[model_name]


def get_model_nbits(model) -> int:
    """Extract number of bits (capacity) for a loaded model."""
    if hasattr(model, "embedder") and hasattr(model.embedder, "nbits"):
        return model.embedder.nbits
    elif hasattr(model, "nbits"):
        return model.nbits
    return 256


# --- Image Watermarking Functions ---

def embed_image(
    image: Image.Image,
    model_name: str,
    scaling_w: float,
    use_custom_msg: bool,
    custom_msg_str: str,
):
    """Embed watermark into an image."""
    if image is None:
        return None, "Please upload an image first."

    try:
        model = get_model(model_name)
        model.blender.scaling_w = float(scaling_w)

        nbits = get_model_nbits(model)

        # Handle message creation
        if use_custom_msg and custom_msg_str.strip():
            msg_bits = [int(b) for b in custom_msg_str.strip() if b in ("0", "1")]
            if len(msg_bits) < nbits:
                # Pad with zeros or truncate
                msg_bits = (msg_bits + [0] * nbits)[:nbits]
            elif len(msg_bits) > nbits:
                msg_bits = msg_bits[:nbits]
            msg_tensor = torch.tensor([msg_bits], dtype=torch.float32, device=DEVICE)
        else:
            msg_bits = [random.choice([0, 1]) for _ in range(nbits)]
            msg_tensor = torch.tensor([msg_bits], dtype=torch.float32, device=DEVICE)

        msg_str = "".join(str(b) for b in msg_bits)

        # Convert PIL Image to Tensor [1, 3, H, W]
        img_tensor = T.ToTensor()(image.convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model.embed(img_tensor, msgs=msg_tensor, is_video=False)

        watermarked_tensor = outputs["imgs_w"][0].cpu()
        watermarked_pil = T.ToPILImage()(watermarked_tensor.clamp(0, 1))

        info_text = f"Successfully watermarked with model '{model_name}'.\n" \
                    f"Message ({len(msg_bits)} bits): {msg_str}"

        return watermarked_pil, info_text, msg_str
    except Exception as e:
        return None, f"Error embedding watermark: {str(e)}", ""


def detect_image(image: Image.Image, model_name: str):
    """Detect watermark in an image."""
    if image is None:
        return "Please upload an image first."

    try:
        model = get_model(model_name)
        img_tensor = T.ToTensor()(image.convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            detected = model.detect(img_tensor, is_video=False)

        preds = detected["preds"]
        # Extract binary bits (skipping detection bit at index 0)
        msg_bits = (preds[0, 1:] > 0).float().cpu().numpy().astype(int)
        msg_str = "".join(str(b) for b in msg_bits)

        # Detection probability score if available
        det_score = torch.sigmoid(preds[0, 0]).item() if preds.shape[1] > 0 else None
        score_text = f"\nDetection Score: {det_score:.4f}" if det_score is not None else ""

        return f"Extracted Binary Message ({len(msg_bits)} bits):\n{msg_str}{score_text}"
    except Exception as e:
        return f"Error detecting watermark: {str(e)}"


# --- Video Watermarking Functions ---

def embed_video(
    video_path: str,
    model_name: str,
    scaling_w: float,
    watermark_audio: bool,
):
    """Embed watermark into a video (and optionally its audio track)."""
    if video_path is None or not os.path.exists(video_path):
        return None, "", "Please upload a video file first."

    try:
        model = get_model(model_name)
        model.blender.scaling_w = float(scaling_w)

        # Read video
        video, audio, info = read_video_file(video_path)
        fps = info.get("video_fps", 30)
        sample_rate = info.get("audio_fps", 44100) if "audio_fps" in info else 16000

        video = video.float() / 255.0
        video = video.to(DEVICE)

        with torch.no_grad():
            outputs = model.embed(video, is_video=True, lowres_attenuation=True)

        video_w = outputs["imgs_w"].cpu()
        video_msgs = outputs["msgs"][0].cpu().numpy().astype(int)
        v_msg_str = "".join(str(b) for b in video_msgs)

        audio_msg_str = ""
        # Handle audio watermarking
        if watermark_audio and "audio_fps" in info and is_audioseal_installed and audio.numel() > 0:
            audio = audio.float()
            audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)

            if audio_16k.shape[0] > 1:
                audio_16k_mono = torch.mean(audio_16k, dim=0, keepdim=True)
            else:
                audio_16k_mono = audio_16k

            audio_16k_mono_batched = audio_16k_mono.unsqueeze(0).to(DEVICE)
            audio_model = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)

            with torch.no_grad():
                audio_msg_tensor = torch.randint(
                    0, 2, (1, audio_model.msg_processor.nbits), device=DEVICE
                )
                watermark = audio_model.get_watermark(
                    audio_16k_mono_batched, 16000, message=audio_msg_tensor
                )

            audio_16k_w = (audio_16k_mono_batched + watermark).squeeze(0).cpu()

            if audio_16k.shape[0] > 1:
                audio_16k_w = audio_16k_w.repeat(audio_16k.shape[0], 1)

            audio_w = torchaudio.transforms.Resample(16000, sample_rate)(audio_16k_w)
            a_bits = audio_msg_tensor[0].cpu().numpy().astype(int)
            audio_msg_str = "".join(str(b) for b in a_bits)
        elif "audio_fps" in info and audio.numel() > 0:
            audio_w = audio
        else:
            audio_w = None

        # Output file path
        out_dir = tempfile.mkdtemp()
        output_filename = os.path.join(out_dir, "watermarked_video.mp4")

        if audio_w is None:
            # Create a dummy audio tensor with zeros if no audio is present
            num_frames = video_w.shape[0]
            audio_samples = int((num_frames / fps) * sample_rate)
            audio_w = torch.zeros((1, max(audio_samples, 100)), dtype=torch.float32)

        save_video_audio_to_mp4(
            video_tensor=video_w,
            audio_tensor=audio_w,
            fps=int(fps),
            audio_sample_rate=int(sample_rate),
            output_filename=output_filename,
        )

        status_text = f"Video successfully watermarked using '{model_name}'!"
        if watermark_audio and is_audioseal_installed and audio_msg_str:
            status_text += "\nAudio track watermarked with AudioSeal."

        msg_display = f"Video Message ({len(v_msg_str)} bits):\n{v_msg_str}"
        if audio_msg_str:
            msg_display += f"\n\nAudio Message ({len(audio_msg_str)} bits):\n{audio_msg_str}"

        return output_filename, msg_display, status_text
    except Exception as e:
        return None, "", f"Error watermarking video: {str(e)}"


def detect_video(video_path: str, model_name: str, detect_audio: bool):
    """Detect watermark from a video (and optionally audio)."""
    if video_path is None or not os.path.exists(video_path):
        return "Please upload a video file first."

    try:
        model = get_model(model_name)
        video, audio, info = read_video_file(video_path)
        video = video.float() / 255.0
        video = video.to(DEVICE)

        with torch.no_grad():
            msg_extracted = model.extract_message(video)

        v_bits = msg_extracted[0].cpu().numpy().astype(int)
        v_msg_str = "".join(str(b) for b in v_bits)

        result_text = f"Extracted Video Message ({len(v_msg_str)} bits):\n{v_msg_str}\n"

        if detect_audio and "audio_fps" in info and is_audioseal_installed and audio.numel() > 0:
            sample_rate = info["audio_fps"]
            audio = audio.float()
            if len(audio.shape) == 2:
                audio = audio.unsqueeze(0)
            if audio.shape[1] > 1:
                audio = torch.mean(audio, dim=1, keepdim=True)

            detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
            audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio.to(DEVICE))

            with torch.no_grad():
                result, message = detector.detect_watermark(audio_16k, 16000)

            result_text += f"\nAudio Watermark Detection Result: {result.cpu().numpy()}\n"
            if message is not None:
                a_bits = message[0].cpu().numpy().astype(int)
                a_msg_str = "".join(str(b) for b in a_bits)
                result_text += f"Extracted Audio Message: {a_msg_str}"

        return result_text
    except Exception as e:
        return f"Error detecting watermark in video: {str(e)}"


# --- Gradio UI Layout ---

def build_app():
    title = "🦭 VideoSeal Web UI"
    description = """
    **VideoSeal** provides state-of-the-art invisible watermarking for Images and Videos.
    Choose a tab below to embed or detect watermarks easily.
    """

    model_options = ["videoseal", "pixelseal", "chunkyseal"]

    with gr.Blocks(title="VideoSeal Studio") as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Tabs():
            # TAB 1: Image Watermarking
            with gr.Tab("🖼️ Image Watermarking"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. Embed Watermark")
                        img_input = gr.Image(type="pil", label="Input Image")
                        model_select_img = gr.Dropdown(
                            choices=model_options, value="videoseal", label="Watermark Model"
                        )
                        strength_img = gr.Slider(
                            minimum=0.05, maximum=1.0, value=0.2, step=0.05, label="Watermark Strength (scaling_w)"
                        )

                        use_custom_msg = gr.Checkbox(value=False, label="Specify Custom Binary Message")
                        custom_msg_input = gr.Textbox(
                            label="Custom Binary Message (e.g. 10101...)",
                            placeholder="Type 0s and 1s...",
                            visible=False,
                        )

                        def toggle_custom_msg(val):
                            return gr.update(visible=val)

                        use_custom_msg.change(
                            fn=toggle_custom_msg, inputs=[use_custom_msg], outputs=[custom_msg_input]
                        )

                        embed_img_btn = gr.Button("Embed Watermark", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Watermarked Output")
                        img_output = gr.Image(type="pil", label="Watermarked Image")
                        img_status = gr.Textbox(label="Status & Message Details", interactive=False)

                        embed_img_btn.click(
                            fn=embed_image,
                            inputs=[img_input, model_select_img, strength_img, use_custom_msg, custom_msg_input],
                            outputs=[img_output, img_status, custom_msg_input],
                        )

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 2. Detect / Extract Watermark")
                        detect_img_input = gr.Image(type="pil", label="Watermarked Image to Detect")
                        detect_model_select_img = gr.Dropdown(
                            choices=model_options, value="videoseal", label="Detection Model"
                        )
                        detect_img_btn = gr.Button("Detect Watermark", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Extraction Results")
                        detect_img_output = gr.Textbox(label="Extracted Watermark Results", lines=5, interactive=False)

                        detect_img_btn.click(
                            fn=detect_image,
                            inputs=[detect_img_input, detect_model_select_img],
                            outputs=[detect_img_output],
                        )

            # TAB 2: Video Watermarking
            with gr.Tab("🎬 Video Watermarking"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. Embed Watermark in Video")
                        video_input = gr.Video(label="Input Video (MP4)")
                        model_select_vid = gr.Dropdown(
                            choices=model_options, value="videoseal", label="Watermark Model"
                        )
                        strength_vid = gr.Slider(
                            minimum=0.05, maximum=1.0, value=0.2, step=0.05, label="Watermark Strength (scaling_w)"
                        )
                        wm_audio_chk = gr.Checkbox(
                            value=is_audioseal_installed,
                            label="Watermark Audio Track (AudioSeal required)",
                            interactive=is_audioseal_installed,
                        )

                        embed_vid_btn = gr.Button("Embed Video Watermark", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Watermarked Output")
                        video_output = gr.Video(label="Watermarked Video")
                        vid_msg_output = gr.Textbox(label="Embedded Messages", lines=4, interactive=False)
                        vid_status = gr.Textbox(label="Status", interactive=False)

                        embed_vid_btn.click(
                            fn=embed_video,
                            inputs=[video_input, model_select_vid, strength_vid, wm_audio_chk],
                            outputs=[video_output, vid_msg_output, vid_status],
                        )

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 2. Detect Watermark in Video")
                        detect_vid_input = gr.Video(label="Watermarked Video File")
                        detect_model_select_vid = gr.Dropdown(
                            choices=model_options, value="videoseal", label="Detection Model"
                        )
                        detect_audio_chk = gr.Checkbox(
                            value=is_audioseal_installed,
                            label="Detect Audio Watermark",
                            interactive=is_audioseal_installed,
                        )

                        detect_vid_btn = gr.Button("Detect Video Watermark", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Extraction Results")
                        detect_vid_output = gr.Textbox(
                            label="Extracted Watermark Results", lines=6, interactive=False
                        )

                        detect_vid_btn.click(
                            fn=detect_video,
                            inputs=[detect_vid_input, detect_model_select_vid, detect_audio_chk],
                            outputs=[detect_vid_output],
                        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
