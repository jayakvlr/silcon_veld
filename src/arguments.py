from argparse import ArgumentParser
from swin_falcon_model.utils.utils import set_seed, get_config


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str,
                        default='swin_base_patch4_window7_224_in22k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str,
                        default='repblock', choices=['repattn', 'repblock'])
    parser.add_argument('--scale', type=float, default=0)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--few-shot',  action='store_true')
    parser.add_argument('--shots',   type=int, default=1)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    config = get_config(args.method, args.dataset, args.few_shot)
