from siamese_unet import SiameseUnet
import torch


def main():
    model = SiameseUnet()
    model.to(device='cpu')
    model.eval()

    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    batch_size = 4
    channels = 2  # Real and Imaginary components [cite: 90]
    freq_bins = 128
    time_frames = 128

    output_file = "siamese_unet.onnx"

    single_clip = torch.randn(batch_size, channels, freq_bins, time_frames)
    reference = torch.randn(batch_size, channels, freq_bins, time_frames)
    output = model(single_clip, reference)

    print("Called model on a single input pair")
    print(f"Mixture input shape: {single_clip.shape}")
    print(f"Reference input shape: {reference.shape}")
    print(f"Output shape: {output.shape}")

    torch.onnx.export(
        model,
        (single_clip, reference),
        output_file,
        export_params=True,
        opset_version=9,          # nncase v0.2.0-beta4 works best with opset 9
        input_names=['mixture_ri', 'reference_ri'],
        output_names=['estimated_ri'],
        verbose=False
    )
    print("Export done")


if __name__ == "__main__":
    main()

