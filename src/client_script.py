import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import json

from model import SASRec 
from data_preprocess import preprocess
from utils import evaluate, evaluate_valid, data_partition, WarpSampler

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

# Setup argument parsing for SASRec-specific hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Movies_and_TV')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

def main():
    print("Reuben's Client:",args)
    
    # Initialize NVFlare communication
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    client_no = client_name.split('-')[1]
    print("Client "+client_name)

    text_file_path=f'/home/reuben/reuben_code/Pytorch/LLM-Rec-Sys/FedSasrec/data/amazon/split_data_part{client_no}.txt'
    
    # Check if the file exists
    if not os.path.exists(text_file_path):
        print(f"File {text_file_path} not found. Running preprocess function...")
        preprocess(args.dataset)  # Call preprocess function if file doesn't exist
    dataset = data_partition(f'split_data_part{client_no}')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size

    # Initialize dataloader (WarpSampler for SASRec)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    # Initialize SASRec model
    arg_handler = json.dumps(args.__dict__)

    l = arg_handler.replace('"', '&')[1:-1]
    model = SASRec(usernum, itemnum, l).to(args.device)

    # Xavier initialization
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # Load model if state_dict provided
    if args.state_dict_path:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
        except:
            print('Failed loading state_dict:', args.state_dict_path)

    # Prepare optimizer and loss function
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # Set up summary writer for logging
    summary_writer = SummaryWriter()

    # Federated Learning Loop
    while flare.is_running():
        print("\nbefore recieve")
        input_model = flare.receive()
        print("\nafter recieve")
        print(f"current_round={input_model.current_round}")

        # Load global model params
        model.load_state_dict(input_model.params)
        model.to(args.device)
        
        # Training loop
        steps = args.num_epochs * len(user_train)
        model.train()

        for epoch in range(1, args.num_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

                # Zero gradients and perform backward pass
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                
                # Apply regularization
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                
                loss.backward()
                adam_optimizer.step()

                # Log the loss at intervals
                if step % 100 == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                    global_step = input_model.current_round * steps + epoch * num_batch + step
                    summary_writer.add_scalar(tag="loss_for_each_batch", scalar=loss.item(), global_step=global_step)

        # Evaluate model
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            print('Evaluating...')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(f"Validation NDCG@10: {t_valid[0]}, HR@10: {t_valid[1]}")
            print(f"Test NDCG@10: {t_test[0]}, HR@10: {t_test[1]}")
            model.train()

        # Send the updated model back to the server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps}
        )

        flare.send(output_model)

    # Clean up resources
    sampler.close()
    print("Training Complete")


if __name__ == "__main__":
    main()
