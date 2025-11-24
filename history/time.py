import os
import soundfile as sf

root = "data/HD-Track2-Test"
folders = ["clean", "test"]

def get_total_hours(path):
    total_seconds = 0.0
    for dp, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_path = os.path.join(dp, f)
                try:
                    audio, sr = sf.read(wav_path)
                    total_seconds += len(audio) / sr
                except Exception as e:
                    print("Error reading:", wav_path, e)
    return total_seconds / 3600


if __name__ == "__main__":
    clean_path = os.path.join(root, "clean")
    test_path = os.path.join(root, "test")

    clean_hours = get_total_hours(clean_path)
    test_hours = get_total_hours(test_path)
    total_hours = clean_hours + test_hours

    print(f"Clean hours: {clean_hours:.3f}")
    print(f"Test  hours: {test_hours:.3f}")
    print(f"Total hours: {total_hours:.3f}")
