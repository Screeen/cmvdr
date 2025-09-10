# tests/test_audio_disk_loader.py
import io
import sys
import tempfile
import shutil
import warnings
import unittest
from pathlib import Path
import numpy as np

from cmvdr.data_gen.audio_disk_loader import AudioDiskLoader


class AudioDiskLoaderTests(unittest.TestCase):
    def setUp(self):
        # create a temporary directory for test files
        self.tmpdir = Path(tempfile.mkdtemp()).expanduser().resolve()
        self.sr = 16000
        # small test signals
        self.mono_signal = np.linspace(-0.1, 0.1, 256).astype(np.float32)
        # stereo signal: shape (n_samples, n_channels) for soundfile.write
        self.stereo_signal = np.vstack([self.mono_signal, self.mono_signal * 0.5]).T

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write_wav(self, filename: str, data: np.ndarray, sr: int):
        return AudioDiskLoader.save_audio_file(self.tmpdir / filename, data, sr)

    def test_load_audio_files_invalid_path_type(self):
        with self.assertRaises(ValueError):
            AudioDiskLoader.load_audio_files(12345)  # not str or Path

    def test_load_audio_files_nonexistent_path(self):
        missing = self.tmpdir / "does_not_exist"
        with self.assertRaises(FileNotFoundError):
            AudioDiskLoader.load_audio_files(missing)

    def test_load_single_file_mono_and_fileid_enrichment(self):
        # create a file with a fileid in the name
        fname = "noisy_fileid_42_test.wav"
        p = self.write_wav(fname, self.mono_signal, self.sr)
        out = AudioDiskLoader.load_audio_files(p, fs=self.sr)
        self.assertIn(p.name, out)
        entry = out[p.name]
        self.assertIn("signal", entry)
        self.assertIn("sr", entry)
        self.assertIn("path", entry)
        self.assertIn("fileid", entry)
        self.assertEqual(entry["sr"], self.sr)
        # fileid should be extracted as "42"
        self.assertEqual(entry["fileid"], "42")

    def test_load_single_file_stereo(self):
        fname = "mix_fileid_7.wav"
        p = self.write_wav(fname, self.stereo_signal, self.sr)
        out = AudioDiskLoader.load_audio_files(p, fs=self.sr)
        entry = out[p.name]
        sig = entry["signal"]
        # librosa with mono=False returns shape (channels, samples) for multi-channel
        # accept either (channels, samples) or (samples, channels) depending on backend
        self.assertTrue(sig.ndim in (1, 2))
        self.assertEqual(entry["sr"], self.sr)
        self.assertEqual(entry["path"], p)

    def test_load_directory_only_wav_files_and_skip_non_wav(self):
        # create wav and a txt file
        wav1 = self.write_wav("a_fileid_1.wav", self.mono_signal, self.sr)
        wav2 = self.write_wav("b_fileid_2.wav", self.mono_signal, self.sr)
        txt = self.tmpdir / "ignore.txt"
        txt.write_text("no audio here")
        out = AudioDiskLoader.load_audio_files(self.tmpdir, fs=self.sr)
        # keys should include only the wav files
        self.assertIn(wav1.name, out)
        self.assertIn(wav2.name, out)
        self.assertNotIn("ignore.txt", out)

    def test_load_directory_no_wav_raises(self):
        # create only a non-wav file
        (self.tmpdir / "only.txt").write_text("nothing")
        with self.assertRaises(ValueError):
            AudioDiskLoader.load_audio_files(self.tmpdir)

    def test_find_id_from_filename_and_warnings(self):
        # correct extraction
        fid = AudioDiskLoader.find_id_from_filename("prefix_fileid_123_suffix.wav")
        self.assertEqual(fid, "123")
        # no fileid triggers warning and returns None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fid2 = AudioDiskLoader.find_id_from_filename("no_id_here.wav")
            self.assertIsNone(fid2)
            self.assertTrue(any("No file ID" in str(x.message) for x in w))

    def test_enrich_with_fileid_non_integer_sets_none(self):
        files = {
            "bad_fileid_xyz.wav": {"signal": self.mono_signal, "sr": self.sr, "path": "p"}
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = AudioDiskLoader.enrich_with_fileid(files)
            self.assertIn("bad_fileid_xyz.wav", out)
            self.assertIsNone(out["bad_fileid_xyz.wav"]["fileid"])

    def test_save_audio_file_prints_and_creates_file(self):
        out_file = self.tmpdir / "out.wav"
        buf = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = buf
            AudioDiskLoader.save_audio_file(out_file, self.mono_signal, self.sr)
        finally:
            sys.stdout = old
        self.assertTrue(out_file.exists())
        self.assertIn("Saved audio to", buf.getvalue())

    def test_save_audio_files_valid_and_invalid_args(self):
        # invalid signals_dict type
        with self.assertRaises(ValueError):
            AudioDiskLoader.save_audio_files(["not", "a", "dict"], self.tmpdir, self.sr)
        # invalid output_folder type
        with self.assertRaises(ValueError):
            AudioDiskLoader.save_audio_files({"a": {"clean": self.mono_signal}}, 123, self.sr)

        # valid save: should create one file (noise_cov_est should be skipped)
        signals = {
            "file42": {
                "clean_baby": self.mono_signal,
                "other": self.mono_signal * 0.5,
                "noise_cov_est": np.zeros((2, 2))
            }
        }
        out_folder = self.tmpdir / "out"
        AudioDiskLoader.save_audio_files(signals, out_folder, self.sr, export_list=["clean_baby"])
        # file should exist: output name: f"{name_no_ext}_{key}.wav"
        expected = out_folder / "file42_clean_baby.wav"
        self.assertTrue(expected.exists())

    def test_save_audio_files_with_export_list(self):
        signals = {
            "x.wav": {"a": self.mono_signal, "b": self.mono_signal}
        }
        out_folder = self.tmpdir / "out2"
        # export only 'b'
        AudioDiskLoader.save_audio_files(signals, out_folder, self.sr, export_list=["b"])
        self.assertTrue((out_folder / "x_b.wav").exists())
        self.assertFalse((out_folder / "x_a.wav").exists())


if __name__ == "__main__":
    unittest.main()
