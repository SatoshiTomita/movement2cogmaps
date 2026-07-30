import torch
class mytorch:
    @staticmethod
    def concat(tensor_list: list, dim=0):
        if None in tensor_list:
            return None
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def stack(tensor_list: list, dim=0):
        if None in tensor_list:
            return None
        return torch.stack(tensor_list, dim=dim)

    @staticmethod
    @torch.jit.script
    def softmax(
        inputs: torch.Tensor, dim: int, temperature: torch.Tensor = torch.tensor(1.0)
    ):

        x = inputs - torch.max(inputs.detach(), dim=-1, keepdim=True)[0]
        x = x / temperature

        x = torch.softmax(x, dim=dim)

        if torch.isinf(x).any() or torch.isnan(x).any():
            print("inputs", inputs)
            print("result", x)
            raise ValueError("softmax is inf or nan")

        return x