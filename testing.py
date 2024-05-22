from src.model_utils import load_model, load_video, process_video


def main():
    is_cnn = True
    is_mobilenet = False
    scene = "Foulspiel"

    model, _, config = load_model(is_cnn, is_mobilenet, scene)
    cap, fps = load_video(f"./data/one_game/Test_{scene}.mp4")
    _ = process_video(cap, fps, model, config, is_cnn, is_mobilenet)


if __name__ == "__main__":
    main()
