import torchaudio
import argparse
import os

from speechbrain.inference.separation import SepformerSeparation


def main(separator: str):
    model = SepformerSeparation.from_hparams(source=separator, savedir=f'pretrained_models/{separator.split("/")[-1]}')

    # for custom file, change path
    est_sources = model.separate_file(path='speechbrain/sepformer-wsj02mix/test_mixture.wav') 

    torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple script that reads from an input file, "
                    "uses a specified separator, and writes to an output file."
    )

    parser.add_argument(
        '--separator',
        type=str,
        default='speechbrain/resepformer-wsj02mix',  
        help='Hugging Face model to use as separator'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='The path to the input file to be processed.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='The path to the output file where results will be written.'
    )

    args = parser.parse_args()

    main(separator=args.separator)
