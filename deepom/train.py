#!/usr/bin/env python
import argparse

from deepom.localizer_real_data import DataLocalizer
import timeit
import sys
import numpy
from deepom.config import Config



def train_model(train_size,batch_size,simulated,create_data,epochs,checkpoint_file):
    localizer = DataLocalizer()
    localizer.top_mol_num = train_size
    localizer.checkpoint_file = checkpoint_file if checkpoint_file.endswith(".pickle") else checkpoint_file+".pickle"
    localizer.simulated_images = simulated
    assert localizer.simulated_images
    if create_data:
        train_data, test_data = localizer.prepare_data()
    else:
        train_file_name = ".train_data_"
        if simulated:
            train_file_name+="simulated_"
            
        train_file_name+= "images_512.pickle"  
        
        print(train_file_name)
        try:
            train_data = localizer.pickle_load(Config.GROUND_TRUTH_DIR.with_suffix(train_file_name))
        except FileNotFoundError:
            pass
    localizer.log_wandb_enable = True
    localizer.epochs = epochs
    localizer.train(train_data)
    if localizer.stop_training == True:
        print("Model stopped training early")
    else:
        print("training complete")
#         test_data = localizer.pickle_load(Config.GROUND_TRUTH_DIR.with_suffix(".test_data_simulated_images_512.pickle"))
    
    
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_size", help="the number of samples the model will train on", type=check_positive, default=512)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--simulated",  action="store_true", default=True,help="should the model train on real or simulated data")
    parser.add_argument("batch_size", help="the batch size per step",type=int,default=1)
    parser.add_argument("-d", "--create-data",  action="store_true", default=False,help="should new data be created before training")
    parser.add_argument("checkpoint_name",type=str,default=Config.CHECKPOINT_FILE)
    args = parser.parse_args()
    # parser.add_argument("epochs", type=check_positive, help="number of training epochs", default=10)
    
    train_size = args.train_size
    simulated = args.simulated
    batch_size = args.batch_size
    create_data = args.create_data
    checkpoint_name = args.checkpoint_name
    epochs = 500
    print(f"train size {train_size}")
    print(f"simulated {simulated}")
    print(f"batch size {batch_size}")
    print(f"create_data {create_data}")
    print(f"checkpoint_name {checkpoint_name}")
    train_model(train_size,batch_size,simulated,create_data,epochs,checkpoint_name)

                        
            