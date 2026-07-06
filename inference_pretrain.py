from core.inference.rynnworld_teleop_sft import generate_video_sft
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for RynnWorld-Teleop SFT pretrained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint directory (containing ema_weights.pt). If not set, uses base model.")
    parser.add_argument("--output", type=str, default="results/sft_pretrain",
                        help="Output directory for generated videos")
    parser.add_argument("--model_path", type=str,
                        default=os.environ.get("MODEL_PATH", "pretrained/Wan2.2-TI2V-5B-Diffusers"))
    parser.add_argument("--data_json", type=str,
                        default=os.environ.get("DATA_JSON", "example/example_cases.json"),
                        help="Path to JSON file with video_latent_path and text_embedding_path entries")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to randomly select for inference")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ema", action="store_true",
                        help="Use raw model weights instead of EMA weights")
    args = parser.parse_args()

    generate_video_sft(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_path=args.model_path,
        data_json=args.data_json,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        use_ema=not args.no_ema,
    )
