from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import random
import jsonlines
from base import get_sorted_expressions

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    yml_cfg = OmegaConf.load(yaml_path)
    cli_cfg = OmegaConf.from_cli(args_list)
    cfg = OmegaConf.merge(yml_cfg, cli_cfg)

    samples = [cfg.sample_per_try // 3 + (1 if x < cfg.sample_per_try % 3 else 0)  for x in range (3)]
    tries = cfg.tries
    difficulties = {
        0: lambda li: li[:samples[0]],  # easy
        1: lambda li: li[len(li)//2-samples[1]//2:len(li)//2+(samples[1] - samples[1]//2)],  # medium
        2: lambda li: li[-samples[2]:]  # hard
    }

    data = {}
    with tqdm(total=tries*cfg.sample_per_try) as pbar, jsonlines.open(cfg.output, 'w') as writer:
        while tries:
            # base numbers generated following the official rules of the human game
            base_numbers = [random.randint(1, 4), random.randint(1, 6), random.randint(1, 8), random.randint(1, 12), random.randint(1, 20)]
            sorted_exp = get_sorted_expressions(base_numbers)
            if cfg.sample_per_try > len(sorted_exp):
                continue
            for difficulty in difficulties:
                sorted_exp_slice = difficulties[difficulty](sorted_exp)
                for s in range(min(samples[difficulty], len(sorted_exp_slice))):
                    target = sorted_exp_slice[s][0]
                    mathador_solution = sorted_exp_slice[s][1][0][-1].steps
                    mathador_solution_score = sorted_exp_slice[s][1][0][-1].score
                    simple_solution = sorted_exp_slice[s][1][0][0].steps
                    simple_solution_score = sorted_exp_slice[s][1][0][0].score
                    example = {
                        'target': target,
                        'base_numbers': base_numbers,
                        'mathador_solution': mathador_solution,
                        'mathador_solution_score': mathador_solution_score,
                        'simple_solution': simple_solution if mathador_solution_score != simple_solution_score else None,
                        'simple_solution_score': simple_solution_score if mathador_solution_score != simple_solution_score else None,
                    }
                    writer.write(example)
                    pbar.update(1)
            tries -= 1
