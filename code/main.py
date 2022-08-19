import time
import os
import torch
from configs.loading_configs import loading_args
from functools import reduce
import lib.utils as utils
from lib.load_data import load_data
from model.generic_neural_net import Model

def secondsToStr(t):
    return '%d:%02d:%02d.%03d' % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

start_time = time.time()
print(time.strftime('Start time: %H:%M:%S', time.localtime()))

args = loading_args() # load configs
utils.check_path(args) # check path
args.dataset = load_data(os.path.join(args.datapath, args.dataset_name)) # load data

model = Model(
    # model
    model_configs=args.model_configs[args.model],
    basic_configs={
        'model': args.model,
        # loading data
        'dataset': args.dataset,
        # train configs
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        # loading configs
        'checkpoint_dir': args.checkpoint_dir,
        'model_name': '%s_%s_wd%.0e' % (
            args.dataset_name, args.model, args.weight_decay)
    }
)

if args.task == None:
    print("Please input a task to excuse")
elif args.task =='test':
    model.train(verbose=True, plot=True)
    eva_x, eva_y = model.np2tensor(model.dataset['test'].get_batch())
    eva_diff = model.model(eva_x) - eva_y
    print(len(eva_diff[torch.abs(eva_diff)<0.5])/len(eva_diff))
else:
    raise Exception('No such a task {}'.format(args.task))

end_time = time.time()
print(time.strftime('%H:%M:%S', time.localtime()))
print('Use time: {}'.format(secondsToStr(end_time - start_time)))