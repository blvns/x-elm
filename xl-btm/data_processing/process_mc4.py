from datasets import load_dataset
import os
import argparse
import time
import numpy as np

PATH_TO_CACHE = '/gscratch/scrubbed/blvns/'

SHARD_COUNTS = {
    "af": {"train": 64, "validation": 1},
    "am": {"train": 16, "validation": 1},
    "ar": {"train": 1024, "validation": 4},
    "az": {"train": 256, "validation": 1},
    "be": {"train": 128, "validation": 1},
    "bg": {"train": 1024, "validation": 1},
    "bg-Latn": {"train": 4, "validation": 1},
    "bn": {"train": 512, "validation": 1},
    "ca": {"train": 512, "validation": 1},
    "ceb": {"train": 8, "validation": 1},
    "co": {"train": 8, "validation": 1},
    "cs": {"train": 1024, "validation": 2},
    "cy": {"train": 256, "validation": 1},
    "da": {"train": 1024, "validation": 1},
    "de": {"train": 2048, "validation": 16},
    "el": {"train": 1024, "validation": 2},
    "el-Latn": {"train": 16, "validation": 1},
    "en": {"train": 11264, "validation": 128},
    "eo": {"train": 32, "validation": 1},
    "es": {"train": 2048, "validation": 16},
    "et": {"train": 256, "validation": 1},
    "eu": {"train": 64, "validation": 1},
    "fa": {"train": 1024, "validation": 2},
    "fi": {"train": 1024, "validation": 1},
    "fil": {"train": 64, "validation": 1},
    "fr": {"train": 2048, "validation": 16},
    "fy": {"train": 16, "validation": 1},
    "ga": {"train": 16, "validation": 1},
    "gd": {"train": 16, "validation": 1},
    "gl": {"train": 128, "validation": 1},
    "gu": {"train": 64, "validation": 1},
    "ha": {"train": 8, "validation": 1},
    "haw": {"train": 2, "validation": 1},
    "hi": {"train": 1024, "validation": 2},
    "hi-Latn": {"train": 16, "validation": 1},
    "hmn": {"train": 8, "validation": 1},
    "ht": {"train": 8, "validation": 1},
    "hu": {"train": 1024, "validation": 2},
    "hy": {"train": 128, "validation": 1},
    "id": {"train": 1024, "validation": 4},
    "ig": {"train": 4, "validation": 1},
    "is": {"train": 128, "validation": 1},
    "it": {"train": 1024, "validation": 8},
    "iw": {"train": 1024, "validation": 1},
    "ja": {"train": 1024, "validation": 8},
    "ja-Latn": {"train": 8, "validation": 1},
    "jv": {"train": 8, "validation": 1},
    "ka": {"train": 256, "validation": 1},
    "kk": {"train": 256, "validation": 1},
    "km": {"train": 64, "validation": 1},
    "kn": {"train": 64, "validation": 1},
    "ko": {"train": 1024, "validation": 1},
    "ku": {"train": 16, "validation": 1},
    "ky": {"train": 64, "validation": 1},
    "la": {"train": 64, "validation": 1},
    "lb": {"train": 32, "validation": 1},
    "lo": {"train": 8, "validation": 1},
    "lt": {"train": 512, "validation": 1},
    "lv": {"train": 256, "validation": 1},
    "mg": {"train": 8, "validation": 1},
    "mi": {"train": 4, "validation": 1},
    "mk": {"train": 128, "validation": 1},
    "ml": {"train": 128, "validation": 1},
    "mn": {"train": 128, "validation": 1},
    "mr": {"train": 1024, "validation": 1},
    "ms": {"train": 512, "validation": 1},
    "mt": {"train": 128, "validation": 1},
    "my": {"train": 64, "validation": 1},
    "ne": {"train": 256, "validation": 1},
    "nl": {"train": 1024, "validation": 4},
    "no": {"train": 1024, "validation": 1},
    "ny": {"train": 4, "validation": 1},
    "pa": {"train": 32, "validation": 1},
    "pl": {"train": 1024, "validation": 4},
    "ps": {"train": 16, "validation": 1},
    "pt": {"train": 1024, "validation": 4},
    "ro": {"train": 1024, "validation": 2},
    "ru": {"train": 4096, "validation": 32},
    "ru-Latn": {"train": 32, "validation": 1},
    "sd": {"train": 64, "validation": 1},
    "si": {"train": 64, "validation": 1},
    "sk": {"train": 512, "validation": 1},
    "sl": {"train": 256, "validation": 1},
    "sm": {"train": 4, "validation": 1},
    "sn": {"train": 8, "validation": 1},
    "so": {"train": 64, "validation": 1},
    "sq": {"train": 128, "validation": 1},
    "sr": {"train": 256, "validation": 1},
    "st": {"train": 2, "validation": 1},
    "su": {"train": 4, "validation": 1},
    "sv": {"train": 1024, "validation": 2},
    "sw": {"train": 32, "validation": 1},
    "ta": {"train": 256, "validation": 1},
    "te": {"train": 128, "validation": 1},
    "tg": {"train": 64, "validation": 1},
    "th": {"train": 1024, "validation": 1},
    "tr": {"train": 1024, "validation": 4},
    "uk": {"train": 1024, "validation": 2},
    "und": {"train": 3072, "validation": 32},
    "ur": {"train": 128, "validation": 1},
    "uz": {"train": 32, "validation": 1},
    "vi": {"train": 1024, "validation": 4},
    "xh": {"train": 2, "validation": 1},
    "yi": {"train": 16, "validation": 1},
    "yo": {"train": 2, "validation": 1},
    "zh": {"train": 1024, "validation": 2},
    "zh-Latn": {"train": 8, "validation": 1},
    "zu": {"train": 8, "validation": 1},
}


def main(args):
    tmp_dir = '~/tmp'
    output_dir = {'train': '/gscratch/zlab/blvns/xl-btm/data/mc4_adapt/train', 'validation': '/gscratch/zlab/blvns/xl-btm/data/mc4_adapt/valid'}
    for split in output_dir:
        if not os.path.exists(output_dir[split]): 
            os.makedirs(output_dir[split])

    for lang in args.langs:
        assert lang in SHARD_COUNTS.keys()

        for split in ['validation']: #not doing train
            #load to get full size of dataset
            mc4 = load_dataset('mc4', lang, split=split, cache_dir=PATH_TO_CACHE)

            #calculate which records go in which shards
            split_ids = np.arange(len(mc4))
            split_ids_arr = np.array_split(split_ids, SHARD_COUNTS[lang][split])
            print(len(split_ids_arr))

            for shard_idx, shard_ids in enumerate(split_ids_arr):
                if shard_idx >= 1024:
                	break

                l_name = 'he' if lang == 'iw' else lang
                #skip shards that are already processed
                output_filename = '{}.jsonl'.format(l_name)
                shard_dir = str(shard_idx).zfill(5)
                output_path = os.path.join(output_dir[split], shard_dir)
                if not os.path.exists(output_path): 
                    os.makedirs(output_path)

                output_path = os.path.join(output_path, output_filename)
                if os.path.isfile(output_path):
                    print('skipping shard {}...'.format(shard_idx))
                    continue

                start = time.time()	

                #get individual shards of dataset to process
                #load only indexes for shard
                #shard = load_dataset('mc4', lang, split='{}[{}:{}]'.format(split, shard_ids[0], shard_ids[-1]+1), cache_dir=PATH_TO_CACHE)
                #shard = load_dataset('mc4', lang, split='{}'.format(split), cache_dir=PATH_TO_CACHE)
                shard = mc4.select(shard_ids)

                #add lang code as meta data for each element in shard

                #add index for each element in shard
                element_idxs = list(range(0, len(shard)))
                lang_arr = [l_name]*len(shard)   
                shard = shard.add_column(name="id", column=element_idxs)
                shard = shard.add_column(name="lang", column=lang_arr)

                #DEBUGGING
                print(shard)

                #write out shard as .jsonl file
                shard.to_json(output_path, lines=True, orient='records', \
                	force_ascii=False)

                if split == 'validation':
                    output_path2 = os.path.join(output_dir[split]+'_{}'.format(l_name), shard_dir)
                    if not os.path.exists(output_path2): 
                        os.makedirs(output_path2)
                    output_filename = 'mc4.jsonl'
                    output_path2 = os.path.join(output_path2, output_filename)

                    shard.to_json(output_path2, lines=True, orient='records', \
                    force_ascii=False)


                end = time.time()
                print(str(end-start)+' seconds to process shard '+str(shard_idx))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--langs', nargs='+', required=True)
	args = parser.parse_args()
	print(args)

	main(args)

#EOF
