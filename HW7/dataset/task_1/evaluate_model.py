import os
import torch
import argparse
from model import Generator
from util import get_test_conditions,save_image
from evaluator import evaluation_model

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim=100
c_dim=200
G_times=4
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--test_mode', default="test", type=str)
args = parser.parse_args()
if args.test_mode == "test":
    test_path=os.path.join('test.json')
    generator_path=os.path.join('models/test/batch_size32','epoch157_score0.64.pt')

elif args.test_mode == "new_test":
    test_path=os.path.join('new_test.json')
    generator_path=os.path.join('models/new_test/batch_size32','epoch117_score0.67.pt')

if __name__=='__main__':
    # load testing data conditions
    conditions=get_test_conditions(test_path).to(device)  # (N,24) tensor

    # load generator model
    g_model=Generator(z_dim,c_dim).to(device)
    g_model.load_state_dict(torch.load(generator_path))

    # test
    avg_score=0
    for _ in range(10):
        z = torch.randn(len(conditions), z_dim).to(device)  # (N,100) tensor
        gen_imgs=g_model(z,conditions)
        evaluation = evaluation_model()
        score=evaluation.eval(gen_imgs,conditions)
        print(f'score: {score:.2f}')
        avg_score+=score

    save_image(gen_imgs, os.path.join('gan_results/eval/eval.png'),nrow=8,normalize=True)
    print()
    print(f'avg score: {avg_score/10:.2f}')
