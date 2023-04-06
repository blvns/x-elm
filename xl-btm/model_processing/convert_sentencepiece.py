import sentencepiece as spm
from tqdm import tqdm, trange
import os
import json
from typing import Dict, List, Tuple
import argparse

#Class from Huggingface tokenizers' extraction scripts
#https://github.com/huggingface/tokenizers/blob/main/bindings/python/scripts/sentencepiece_extractor.py
class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models.
    https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        # Get SentencePiece
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in trange(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in tqdm(vocab.keys(), total=sp.GetPieceSize()):
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges


def main(args):

	#convert the model to vocab and merges
	extractor = SentencePieceExtractor(args.spm_path)
	vocab, merges = extractor.extract()

	#write out vocab dict as .json file
	vocab_filepath = os.path.join(args.output_path, "{}-vocab.json".format(args.model_name))
	with open(vocab_filepath, 'w') as f:
		json.dump(vocab, f)

	#write out merges as formatted .txt file
	merges_str = ""
	for a, b in merges:
		merges_str += "{} {}\n".format(a, b)
	merges_str = merges_str.strip()
	merges_filepath = os.path.join(args.output_path, "{}-merges.txt".format(args.model_name))
	with open(merges_filepath, 'w') as f:
		f.write(merges_str)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--spm_path', type=str, required=True)
	parser.add_argument('--model_name', type=str, required=True)
	parser.add_argument('--output_path', type=str, required=True)
	args = parser.parse_args()
	print(args)

	main(args)

#EOF