import streamlit as st
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle as p
from PIL import Image
import torch.optim as optim
from architecture import CifarClassifier
import time


r_class = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5:'dog', 6:'frog', 7: 'horse', 8:'ship', 9: 'truck'}
st.markdown("# An example of training Cifar dataset by ML models")
def get_info():
    data_name = st.sidebar.selectbox("Dataset:", ["cifar-10", "cifar-100"])
    model_name = st.sidebar.selectbox("Model:", ["CNN", "SVM"])
    return data_name, model_name

def parameter_set(model_name):
    params = dict()
    if model_name=="SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        lr = st.sidebar.selectbox("Learning Rate", [3e-4, 1e-4, 1e-3, 1e-2, 0.1])
        batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64, 128])
        optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSProp"])
        epochs = st.sidebar.selectbox("Epoch", [100, 300])
        test_interval = st.sidebar.selectbox("Test Interval", [10, 20, 25, 50])
        distributed = st.sidebar.radio("Ditributed", ["No", "Yes"])
        finish = st.sidebar.radio("Finished Setting", ["No", "Yes"])
        params["lr"] = lr
        params["epochs"] = epochs
        params["batch_size"] = batch_size
        params["optimizer"] = optimizer
        params["distributed"] = distributed
        params["test_interval"] = test_interval
    return params, finish

def get_batchdata(data_name, params):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    if data_name=="cifar-10":
        cifar10_train = datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform) 
        train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=params["batch_size"],shuffle=True)
    elif data_name=="cifar-100":
        cifar100_train = datasets.CIFAR100(root="./dataset", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root="./dataset", train=False, download=True, transform=transform)             
        train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=params["batch_size"],shuffle=True)
    return train_loader, test_dataset

def show_samples(data_name):
    if data_name=="cifar-10":
        train_tmp = datasets.CIFAR10(root="./dataset", train=True, download=True)
    elif data_name=="cifar-100":
        train_tmp = datasets.CIFAR100(root="./dataset", train=True, download=True)
        
    st.markdown(f"## Samples of training data ({data_name})")
    mygrid = [[], []]
    with st.container():
        mygrid[0] = st.columns(2)
    with st.container():
        mygrid[1] = st.columns(2)
    for i, (img, _) in enumerate(train_tmp):
        if i>=4:
            break
        img = np.array(img)
        row = int(i/2)
        col = i%2
        mygrid[row][col].image(img, width=128)
    

def transform_label(label, class_num):
    batch_size = label.shape[0]
    label_new = torch.zeros(batch_size, class_num)
    for i in range(batch_size):
        label_new[i, label[i]] = 1
    return label_new

def test(model, test_dataset):
    model.eval()
    acc_num = 0
    for i, (img, label) in enumerate(test_dataset):
        img = img.unsqueeze(0)
        pred = model(img)
        pred_np = pred.squeeze().detach().cpu().numpy()
        pred_class = np.argmax(pred_np)
        if pred_class==label:
            acc_num+=1
    accuracy = float(acc_num/len(test_dataset))
    # accuracy = float(acc_num/test_num)
    return accuracy
    

def train(model_name, data_name, train_loader, test_dataset, params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if data_name=="cifar-10":
        class_num = 10
    elif data_name=="cifar-100":
        class_num = 100
    if model_name=="CNN":
        model = CifarClassifier(class_num).to(device)
    lr = params['lr']
    batch_size = params['batch_size']
    test_interval = params['test_interval']
    optim_type = params['optimizer']
    if optim_type=="Adam":
        optimizer = optim.Adam(model.parameters(),lr=lr)
    elif optim_type=="SGD":
        optimizer = optim.SGD(model.parameters(),lr=lr)
    elif optim_type=="RMSProp":
        optimizer = optim.RMSProp(model.parameters(), lr=lr)
    total_epochs = params['epochs']
    criterion = torch.nn.CrossEntropyLoss()
    total_num = len(train_loader)*batch_size
    model.train()
    
    test_acc = 0.0
    st.markdown(f"## Start Training...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    last_rows = np.array([1.0])
    chart = st.line_chart(last_rows)
    for epoch in range(total_epochs):
        losses = 0
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = criterion(pred, label)
            losses += loss.item()
            loss.backward()
            optimizer.step()
        train_losses = float(losses/total_num)
        if epoch%test_interval==0:
            test_acc = test(model, test_dataset)
        progress_bar.progress(epoch)
        status_text.text("Test Accuracy: %s" % test_acc)
        chart.add_rows(np.array([train_losses]))
        time.sleep(0.05)

# data_name, model_name = get_info()
# params, finish = parameter_set(model_name)
# show_samples(data_name)
# if finish=="Yes":
#     train_loader, test_dataset = get_batchdata(data_name, params)        
#     train(model_name, data_name, train_loader, test_dataset, params)
            