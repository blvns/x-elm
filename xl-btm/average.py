import torch
from pathlib import Path
from tqdm.auto import tqdm
import argparse

def average(models, weights=None):
        state_dicts = [model['model'] for model in models]
        with torch.no_grad():
            merged = {}
            for key in state_dicts[0]:
                merged[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(state_dicts, weights)]), axis=0)
            return merged

def main(args):
    #default to equal weighting if none is given
    if args.weights: weights = [float(x) for x in args.weights.split(',')]
    else: weights = [float(1/len(args.expert_dirs)) for _ in args.expert_dirs]
    print(weights)

    experts = [Path(x) for x in args.expert_dirs]
    print(experts)
    experts = [torch.load(e / 'consolidated.pt') for e in tqdm(experts)]
    merged_expert = experts[0].copy()
    merged_expert['model'] = average(experts, weights=weights)
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)

    torch.save(merged_expert, args.output_dir / 'consolidated.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expert-dirs", type=Path, nargs='+')
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()
    main(args)