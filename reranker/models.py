import torch
from transformers import AutoModel


class DefaultSelectionModel(torch.nn.Module):
    def __init__(self, lmtype: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(lmtype)
        self.linear = torch.nn.Linear(768, 1)  # TOOD: fix hard-coding

    def forward(self, ids, masks, return_hidden: bool = False, average_hidden: bool = False, return_all_hidden: bool = False):
        if average_hidden or return_all_hidden:
            output = self.model(ids, masks, output_hidden_states=True)
            cls_output = output[0][:, 0]
            if average_hidden:
                layer_output = torch.stack([layer_output[:, 0] for layer_output in output[2][1:]]).mean(0)  # except for word embedding
                assert cls_output.shape == layer_output.shape
                repr_output = layer_output
            elif return_all_hidden:
                layer_output = [layer_output[:, 0] for layer_output in output[2][1:]]  # 12 * (bs,768)
                assert cls_output.shape == layer_output[0].shape and len(list(set([e.shape for e in layer_output]))) == 1
                repr_output = layer_output
        else:
            cls_output = self.model(ids, masks)[0][:, 0]
            repr_output = cls_output

        final_output = self.linear(cls_output)

        if not return_hidden:
            return final_output
        else:
            return final_output, repr_output
