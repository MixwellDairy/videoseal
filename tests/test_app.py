# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import torch
import torchvision
from PIL import Image

import videoseal.utils.display
import app


class TestAppBackend(unittest.TestCase):
    def test_text_to_bits_and_bits_to_text(self):
        text = "Hello VideoSeal!"
        nbits = 256
        bits = app.text_to_bits(text, nbits)
        self.assertEqual(len(bits), nbits)

        decoded = app.bits_to_text(bits)
        self.assertEqual(decoded, text)

    def test_image_embedding_and_detection(self):
        # Create a test image
        img = Image.new("RGB", (128, 128), color="blue")

        # Test image embedding with secret text
        secret_text = "TopSecret"
        watermarked_pil, info_text, msg_str = app.embed_image(
            image=img,
            model_name="videoseal",
            scaling_w=0.2,
            msg_type="Secret Text",
            secret_text=secret_text,
            custom_msg_str="",
        )

        self.assertIsNotNone(watermarked_pil)
        self.assertIn("Successfully watermarked", info_text)
        self.assertEqual(len(msg_str), 256)

        # Test image detection
        detect_result = app.detect_image(
            image=watermarked_pil,
            model_name="videoseal"
        )
        self.assertIn("Watermark Detected:", detect_result)
        self.assertIn("Extracted Secret Text:", detect_result)

    def test_video_embedding_and_detection(self):
        video_tensor = torch.rand((5, 3, 64, 64))
        audio_tensor = torch.zeros((1, 16000))
        test_video_path = "/tmp/test_input_video.mp4"
        videoseal.utils.display.save_video_audio_to_mp4(
            video_tensor=video_tensor,
            audio_tensor=audio_tensor,
            fps=10,
            audio_sample_rate=16000,
            output_filename=test_video_path
        )

        # Test video embedding with secret text
        secret_text = "VidSecret"
        out_path, msg_display, status = app.embed_video(
            video_path=test_video_path,
            model_name="videoseal",
            scaling_w=0.2,
            watermark_audio=False,
            msg_type="Secret Text",
            secret_text=secret_text,
            custom_msg_str="",
        )

        self.assertIsNotNone(out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertIn("Video Binary Message", msg_display)

        # Test video detection
        detect_result = app.detect_video(
            video_path=out_path,
            model_name="videoseal",
            detect_audio=False
        )
        self.assertIn("Watermark Detected:", detect_result)
        self.assertIn("Extracted Secret Text:", detect_result)


if __name__ == "__main__":
    unittest.main()
