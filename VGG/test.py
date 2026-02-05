import model
import torch

if __name__ == "__main__":
    mod = model.vgg11(num_classes=5)
    print(mod)

    dummy_input = torch.randn(2,3,448,448)

    try:
        output = mod(dummy_input)
        print(f'input shape: {dummy_input.shape}')
        print(f'output shape: {output.shape}')
    except Exception as e:
        print(f'error: {e}')
