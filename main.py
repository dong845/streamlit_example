from dashboard import get_info, parameter_set, show_samples, train, get_batchdata, distributed_train
import torch.multiprocessing as mp
from configs import args_dict
import torch

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_name, model_name = get_info()
    params, finish = parameter_set(model_name)
    if finish=="Yes":
        show_samples(data_name)
        if params["distributed"]=="No":      
            train_loader, test_dataset = get_batchdata(data_name, params, 0, 0)
            train(model_name, data_name, train_loader, test_dataset, params)
        elif params["distributed"]=="Yes" and device.type=="cpu":    # not work, streamlit got stuck
            train_loader, test_dataset = get_batchdata(data_name, params, 0, 0)
            num_processes = 2
            processes = []
            for rank in range(num_processes):
                p = mp.Process(target=train, args=(model_name, data_name, train_loader, test_dataset, params,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()     
        # else:                  # work with gpu for distributed
        #     args_dict["data"] = data_name
        #     args_dict["model"] = model_name
        #     for key in params:
        #         args_dict[key] = params[key]
        #     mp.spawn(distributed_train, nprocs=args_dict["gpus"], args=(args_dict,)) 
