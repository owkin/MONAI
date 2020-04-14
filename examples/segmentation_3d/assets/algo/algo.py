import monai
from monai.data import sliding_window_inference

import torch

import substratools as tools


class MonaiAlgo(tools.algo.Algo):
    device = torch.device("cpu")

    def _get_model(self):
        return monai.networks.nets.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)

    def train(self, X, y, models, rank):
        # create UNet, DiceLoss and Adam optimizer
        model = self._get_model()
        loss_function = monai.losses.DiceLoss(do_sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        # start a typical PyTorch training
        model.train()
        for inputs, labels in zip(X, y):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        return model

    def predict(self, X, model):
        y_pred = []
        model.eval()
        with torch.no_grad():
            for inputs, metadata in X:
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
                outputs = (outputs.sigmoid() >= 0.5).float()
                y_pred.append((outputs, metadata))
        return y_pred

    def load_model(self, path):
        model = self._get_model()
        model.load_state_dict(torch.load(path))
        return model

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)


if __name__ == '__main__':
    tools.algo.execute(MonaiAlgo())
