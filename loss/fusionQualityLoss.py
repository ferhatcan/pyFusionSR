import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as tvF
import kornia as kn

from loss.ILoss import ILoss

class FusionQualityLoss(ILoss):
    def __init__(self, args):
        super(FusionQualityLoss, self).__init__()
        self.qLoss = FusionQualityMetric(args)

    def forward(self, data: dict) -> list:
        # data["result"][0] = kn.normalize_min_max(data["result"][0])
        normalLoss = self.qLoss(data)

        # inplace 1 - overallQscore_abf
        for i in range(len(normalLoss)):
            normalLoss[i] = torch.add(normalLoss[i], -1)
            normalLoss[i] = torch.mul(normalLoss[i], -1)

        return normalLoss


class FusionQualityEdgeLoss(ILoss):
    def __init__(self, args):
        super(FusionQualityEdgeLoss, self).__init__()
        self.qLoss = FusionQualityMetric(args)

    def forward(self, data: dict) -> list:
        # data["result"][0] = kn.normalize_min_max(data["result"][0])
        edgeData = {"result": [], "gts": []}
        for inp in data["result"]:
            edgeData["result"].append(kn.normalize_min_max(kn.sobel(inp)))
            mean = edgeData["result"][-1].mean()
            std = edgeData["result"][-1].std()
            edgeData["result"][-1][edgeData["result"][-1] <= mean + 1.2 * std] = 0
        for inp in data["gts"]:
            edgeData["gts"].append(kn.normalize_min_max(kn.sobel(inp)))
            mean = edgeData["gts"][-1].mean()
            std = edgeData["gts"][-1].std()
            edgeData["gts"][-1][edgeData["gts"][-1] <= mean + 1.2 * std] = 0

        edgeLoss = self.qLoss(edgeData)

        for i in range(len(edgeLoss)):
            loss = edgeLoss[i]
            loss = torch.pow(loss, 0.7)
            edgeLoss[i] = loss
        # there is bug in the following line
        # edgeLoss_tmp = [torch.pow(edgeLoss[i], torch.tensor(0.7)) for i in range(len(edgeLoss))]

        # debugging
        # data["edges"] = edgeData
        # debugging

        # inplace 1 - overallQscore_abf
        for i in range(len(edgeLoss)):
            edgeLoss[i] = torch.add(edgeLoss[i], -1)
            edgeLoss[i] = torch.mul(edgeLoss[i], -1)

        return edgeLoss


class FusionQualityMetric(ILoss):
    def __init__(self, args):
        super(FusionQualityMetric, self).__init__()
        self.loss_type = 'Fusion'

        self.WINDOW_SIZE = 8
        self.out_size = tuple((torch.tensor(args.hr_shape) // self.WINDOW_SIZE))

        self.unfold = nn.Unfold(kernel_size=(self.WINDOW_SIZE, self.WINDOW_SIZE), stride=self.WINDOW_SIZE)
        self.fold = nn.Fold(output_size=self.out_size, kernel_size=(1, 1), stride=1)
        self.convert1CH = transforms.Grayscale(num_output_channels=1)
        self.toPIL = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

    def forward(self, data: dict) -> list:
        assert "gts" in data and "result" in data, "gts Type should be a dict and contains \"gts\" and \"result\" keys"
        assert len(data["result"]) == 1, "there should be 1 result to calculate loss"
        # a --> EO, b --> IR, f --> Result
        if data["gts"][0].shape[1] == 1: # if input[1] is IR image with 1 channel
            b = data["gts"][0]
            a = data["gts"][1]
        else:
            b = data["gts"][1]
            a = data["gts"][0]
        a = torch.zeros_like(b)
        f = torch.zeros_like(b)
        # print(b.max(), b.min(), a.max(), a.min(), f.max(), f.min())

        if data["gts"][0].shape[1] == 3:  # if input[1] is IR image with 1 channel
            for i in range(data["gts"][0].shape[0]):
                pil_f = self.toPIL(data["gts"][0][i, ...].cpu().squeeze())
                a[i, ...] = self.toTensor(self.convert1CH(pil_f))
        elif data["gts"][1].shape[1] == 3:  # if input[1] is IR image with 1 channel
            for i in range(data["gts"][1].shape[0]):
                pil_f = self.toPIL(data["gts"][1][i, ...].cpu().squeeze())
                a[i, ...] = self.toTensor(self.convert1CH(pil_f))
        else:
            a = data["gts"][1]
        if data["result"][0].shape[1] == 3:
            for i in range(data["result"][0].shape[0]):
                pil_f = self.toPIL(data["result"][0][i, ...].cpu().squeeze())
                f[i, ...] = self.toTensor(self.convert1CH(pil_f))
        else:
            f = data["result"][0]

        # f1 = f.clone()
        #
        # if f.shape[1] == 3:
        #     for i in range(f.shape[0]):
        #         pil_f = self.toPIL(f[i, ...].cpu().squeeze())
        #         f1[i, ...] = self.toTensor(self.convert1CH(pil_f))
        #
        # f1.to(self.device)
        # Calculate means
        umean_a = self._calculate_mean(a)
        umean_b = self._calculate_mean(b)
        umean_f = self._calculate_mean(f)
        # umean_f1 = self._calculate_mean(f1)

        # Calculate variances
        sigma_a = self._calculate_variance(a, umean_a, a, umean_a)
        sigma_b = self._calculate_variance(b, umean_b, b, umean_b)
        sigma_f = self._calculate_variance(f, umean_f, f, umean_f)
        sigma_af = self._calculate_variance(a, umean_a, f, umean_f)
        # sigma_bf = self._calculate_variance(b, umean_b, f1, umean_f1)
        sigma_bf = self._calculate_variance(b, umean_b, f, umean_f)

        ## Calculate Q0 parts
        # part 1-correlation coefficient
        corrCoeff_af = self._calculte_correlation_coefficient(sigma_a, sigma_f, sigma_af)
        corrCoeff_bf = self._calculte_correlation_coefficient(sigma_b, sigma_f, sigma_bf)
        # part 2-luminance distortion
        lumiDist_af = self._calculate_luminance_distortion(umean_a, umean_f)
        # lumiDist_bf = self._calculate_luminance_distortion(umean_b, umean_f1)
        lumiDist_bf = self._calculate_luminance_distortion(umean_b, umean_f)
        # part 3-contrast distortion
        contDist_af = self._calculate_contrast_distortion(sigma_a, sigma_f)
        contDist_bf = self._calculate_contrast_distortion(sigma_b, sigma_f)

        Q0_af = self._calculate_Q0(corrCoeff_af, lumiDist_af, contDist_af)
        Q0_bf = self._calculate_Q0(corrCoeff_bf, lumiDist_bf, contDist_bf)

        lambda_af, lambda_bf= self._calculate_lambda(sigma_a, sigma_b)

        Q0 = Q0_af * lambda_af + Q0_bf * lambda_bf
        Q0 = Q0 * self._calculate_weighted_quality_index(sigma_a, sigma_b)

        

        # print(Q0.sum())
        overallQscore_abf = Q0.sum() #/ torch.prod(torch.tensor(Q0.shape), 0)
        if overallQscore_abf > 1:
            print(f'loss becomes {overallQscore_abf}')
        # # inplace 1 - overallQscore_abf
        # overallQscore_abf = torch.add(overallQscore_abf, -1)
        # overallQscore_abf = torch.mul(overallQscore_abf, -1)

        return [overallQscore_abf]

    def _calculate_mean(self, x):
        uinput_x = self.unfold(x)
        umean_x = torch.mean(uinput_x, dim=1)
        return umean_x

    def _calculate_variance(self, x, umean_x, y, umean_y):
        uinput_x = self.unfold(x)
        uinput_y = self.unfold(y)
        usub_x = uinput_x - umean_x.unsqueeze(dim=1)
        usub_y = uinput_y - umean_y.unsqueeze(dim=1)
        sigma_xy = self.fold(((usub_x * usub_y) / (self.WINDOW_SIZE * self.WINDOW_SIZE - 1)).sum(dim=1).unsqueeze(dim=1))
        return sigma_xy

    @staticmethod
    def _calculte_correlation_coefficient(sigma_x, sigma_y, sigma_xy):
        corrCoeff_xy = sigma_xy / (torch.sqrt(sigma_x) * torch.sqrt(sigma_y) + 1e-8)
        return corrCoeff_xy

    def _calculate_luminance_distortion(self, umean_x, umean_y):
        mean_x = self.fold(umean_x.unsqueeze(dim=1))
        mean_y = self.fold(umean_y.unsqueeze(dim=1))
        lumiDist = (2 * mean_x * mean_y) / (mean_x * mean_x + mean_y * mean_y + 1e-8)
        return lumiDist

    @staticmethod
    def _calculate_contrast_distortion(sigma_x, sigma_y):
        contDist = (2 * torch.sqrt(sigma_x) * torch.sqrt(sigma_y)) / (sigma_x + sigma_y + 1e-8)
        return contDist

    @staticmethod
    def _calculate_Q0(corrCoeff, lumiDist, contDist_af):
        Q0 = corrCoeff * lumiDist * contDist_af
        Q0[Q0 != Q0] = 0
        #overallQscore = Q0.sum() / torch.prod(torch.tensor(Q0.shape), 0)
        return Q0

    @staticmethod
    def _calculate_lambda(sigma_x, sigma_y):
        lambda_1 = sigma_x / (sigma_x + sigma_y + 1e-8)
        lambda_2 = 1 - lambda_1
        return lambda_1, lambda_2

    @staticmethod
    def _calculate_weighted_quality_index(sigma_x, sigma_y):
        CW = torch.max(sigma_x, sigma_y)
        cW = CW / CW.sum()
        #print(cW.sum())
        return cW
