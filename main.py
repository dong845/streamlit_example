from dashboard import get_info, parameter_set, show_samples, get_batchdata, train

if __name__ == '__main__':
    data_name, model_name = get_info()
    params, finish = parameter_set(model_name)
    if finish=="Yes":
        show_samples(data_name)
        train_loader, test_dataset = get_batchdata(data_name, params)        
        train(model_name, data_name, train_loader, test_dataset, params)
