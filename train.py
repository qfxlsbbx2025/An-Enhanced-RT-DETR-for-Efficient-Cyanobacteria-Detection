from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR("weight.pt") 
    
    train_params = {
        'data': 'emds7-dataset.yaml', 
    }
    model.train(**train_params)