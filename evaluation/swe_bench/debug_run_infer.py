import os

import toml

from openhands.core.config import get_llm_config_arg, get_parser


def debug_config():
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    args, _ = parser.parse_known_args()

    print('\nDebugging config loading:')
    print('1. Checking if config file exists...')
    if not os.path.exists(args.llm_config):
        print(f'Config file not found at: {args.llm_config}')
        return

    print('2. Reading raw TOML content...')
    try:
        with open(args.llm_config, 'r') as f:
            raw_config = toml.load(f)
            print('Raw config:', raw_config)
    except Exception as e:
        print(f'Error reading TOML: {e}')
        return

    print('\n3. Attempting to load through get_llm_config_arg...')
    try:
        llm_config = get_llm_config_arg(args.llm_config)
        print('Loaded config:', llm_config)
        if llm_config is None:
            print('WARNING: get_llm_config_arg returned None!')

        # Try to access the log_completions attribute
        if hasattr(llm_config, 'log_completions'):
            print('log_completions value:', llm_config.log_completions)
        else:
            print('No log_completions attribute found!')

    except Exception as e:
        print(f'Error in get_llm_config_arg: {e}')
        import traceback

        print(traceback.format_exc())


if __name__ == '__main__':
    debug_config()
