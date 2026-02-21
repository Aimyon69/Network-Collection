import model
import torch

if __name__ == '__main__':
    dummy_input = torch.randn(2,3,224,224)

    model = model.GoogLeNet(num_classes=1000)

    model.train()
    aux1,aux2,res = model(dummy_input)
    print('aux1 shape: ',aux1.shape)
    print('aux2 shape: ',aux2.shape)
    print('res shape: ',res.shape)

    print('----------------------------')

    model.eval()
    res = model(dummy_input)
    print('res shape: ',res.shape)