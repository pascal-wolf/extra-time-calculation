import numpy as np


from src.model_utils import (
    load_model,
    process_video,
    load_video,
)


def main():
    is_cnn = True
    is_mobilenet = False
    scene = "Foulspiel"

    model, np_name, config = load_model(is_cnn, is_mobilenet, scene)
    cap, fps = load_video(f"./data/one_game/Test_{scene}.mp4")
    preds = process_video(cap, fps, model, config, is_cnn, is_mobilenet)

    # Save your predictions to a numpy file
    np.save(np_name, preds)


if __name__ == "__main__":
    main()
