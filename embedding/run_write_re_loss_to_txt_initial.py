import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', type=str, default='')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--reference_model', type=str, default='cm_cd_lpips')
parser.add_argument('--reference_bs', type=str, default='2')
parser.add_argument('--filePath', type=str, default='flickr30k')
parser.add_argument('--num_iter', type=int, default=80)
parser.add_argument('--write_path', type=str, default='./result')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument("--use_random_init", action="store_true", help="The path of dev set.")
parser.add_argument('--model_type', type=str, default='')
args = parser.parse_args()


name_list = os.listdir(args.filePath)
name_list = [name for name in name_list if name.lower().endswith('.png')]

counter = 0
for name in name_list:
    print(name)
    if counter ==args.num_samples:
        break
    input_content = args.filePath + name
    print("input_content:",input_content)
    name = name[:-4]

    cmd = 'CUDA_VISIBLE_DEVICES={} python embedding_tracer_initial.py --model_type {} --input_selection_name {} --write_path {} --name {} \
            --bs 1 --num_iter {}'.format(args.gpu,args.model_type,input_content,args.write_path, name, args.num_iter)
    
    counter = counter + 1

    os.system(cmd)