import time 
output_path_def = '/groups/astro/kpr279/test_folder'

def main():
    # take cmd line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=True, action='store_true')
    parser.add_argument('--output_path', type=str, default=output_path_def)
    parser.add_argument('--msg', type=str, default='hello world')
    parser.add_argument('--Nexp', type=int, default=0)
    args = parser.parse_args()

    if args.test:
        print('test mode')
        output_path = '/groups/astro/kpr279/test_folder'
    else:
        output_path = args.output_path

    # save text file saying "hello world"
    with open(f'{output_path}/hello_world.txt', 'w') as f:
        f.write(args.msg)
    
    print(f'Nexp = {args.Nexp}')
    time.sleep(1) # wait 10 seconds
    
if __name__ == '__main__':
    main()
