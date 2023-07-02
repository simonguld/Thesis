
output_path = '/groups/astro/kpr279/test_folder'

def main():
    # save text file saying "hello world"
    with open(f'{output_path}/hello_world.txt', 'w') as f:
        f.write('hello world')
    
if __name__ == '__main__':
    main()
