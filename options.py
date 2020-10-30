# It includes input arguments and options used in system
from configparser import ConfigParser
import importlib
import datetime

class options:
    # @todo some arguments should be tidied. Each argument type should know only their responsibilty range
    # @todo titles should be arranged according to class names.
    def __init__(self, config_file_name):
        self.config = ConfigParser()
        self.config.read(config_file_name)

        self.argsCommon = ParseCommons(self.config)
        self.argsDataset = ParseDataset(self.config)
        self.argsModel = ParseModel(self.config)
        self.argsLoss = ParseLoss(self.config)
        self.argsBenchmark = ParseBenchmark(self.config)
        self.argsExperiment = ParseExperiment(self.config)
        self.argsLog = ParseLog(self.config)
        self.argsMethod = ParseMethod(self.config)

class ParseCommons:
    def __init__(self, config: ConfigParser):
        self.experiment_name    = config["DEFAULT"]["experiment_name"]
        self.generateNew        = config["DEFAULT"].getboolean("generate_new_experiment")
        self.device             = config["HARDWARE"]["device"]
        self.seed               = int(config["HARDWARE"]["seed"])
        self.n_GPUs             = int(config["HARDWARE"]["n_GPUs"])
        self.precision          = config["HARDWARE"]["precision"]
        self.scale              = int(config["DATASET"]["scale"])
        self.model              = config["MODEL"]["model"]
        self.hr_shape           = list(map(int, config["DATASET"]["hr_shape"].split(',')))
        self.batch_size         = int(config["DATASET"]["batch_size"])
        if self.generateNew:
            self.experiment_save_path = "runs/" + self.model + "x{}".format(self.scale) + "/" \
            + self.experiment_name + "_" + datetime.datetime.now().strftime('%Y-%m-%d-hour%H')
        else:
            self.experiment_save_path = "runs/" + self.model + "x{}".format(self.scale) + "/" \
                                        + self.experiment_name


class ParseDataset(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseDataset, self).__init__(config)

        self.train_set_paths    = config["DATASET"]["train_set_paths"].split(',')
        self.test_set_paths     = config["DATASET"]["test_set_paths"].split(',')
        self.rgb_range          = int(config["DATASET"]["rgb_range"])
        self.batch_size         = int(config["DATASET"]["batch_size"])
        self.scale              = int(config["DATASET"]["scale"])
        self.include_noise      = config["DATASET"].getboolean("include_noise")
        self.noise_sigma        = float(config["DATASET"]["noise_sigma"])
        self.noise_mean         = float(config["DATASET"]["noise_mean"])
        self.include_blur       = config["DATASET"].getboolean("include_blur")
        self.blur_radius        = float(config["DATASET"]["blur_radius"])
        self.normalize          = config["DATASET"]["normalize"]
        self.random_flips       = config["DATASET"].getboolean("random_flips")
        self.channel_number     = int(config["DATASET"]["channel_number"])
        self.n_colors           = int(config["DATASET"]["n_colors"])
        self.hr_shape           = list(map(int, config["DATASET"]["hr_shape"].split(',')))
        self.downgrade          = config["DATASET"]["downgrade"]
        self.validation_size    = float(config["DATASET"]["validation_size"])
        self.shuffle_dataset    = config["DATASET"].getboolean("shuffle_dataset")


class ParseModel(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseModel, self).__init__(config)

        self.model = config["MODEL"]["model"]
        self.self_ensemble = config["MODEL"].getboolean("self_ensemble")

        try:
            getattr(self, 'parse'+self.model)(config)
        except:
            self.parseDefault(config)

    def parseDefault(self):
        self.n_resblocks = 32
        self.n_feats = 256
        self.res_scale = 0.1
        self.emb_dimension = 16
        self.filterSize = 17
        self.pretrained = True

    def parsePFF(self, config: ConfigParser):
        self.emb_dimension = int(config["PFF"]["emb_dimension"])
        self.filterSize = int(config["PFF"]["filterSize"])
        self.pretrained = config["PFF"].getboolean("pretrained")

    def parseEDSR(self, config: ConfigParser):
        self.n_resblocks = int(config["EDSR"]["n_resblocks"])
        self.n_feats = int(config["EDSR"]["n_feats"])
        self.res_scale = float(config["EDSR"]["res_scale"])


    def parseEncoderDecoder(self, config: ConfigParser):
        self.type = config['ENCODERDECODER']['type']
        self.ir_pretrained_weights = config['ENCODERDECODER'][
            'ir_pretrained_weights'] if config.has_option('ENCODERDECODER', 'ir_pretrained_weights') else None
        self.eo_pretrained_weights = config['ENCODERDECODER'][
            'eo_pretrained_weights'] if config.has_option('ENCODERDECODER', 'eo_pretrained_weights') else None
        self.ir_channel_number = int(config['ENCODERDECODER']['ir_channel_number']) if config.has_option(
            'ENCODERDECODER', 'ir_channel_number') else 1
        self.eo_channel_number = int(config['ENCODERDECODER']['eo_channel_number']) if config.has_option(
            'ENCODERDECODER', 'eo_channel_number') else 3
        self.output = config['ENCODERDECODER']['output']


class ParseLoss(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseLoss, self).__init__(config)

        self.loss = config["TRAINING"]["loss"]


class ParseBenchmark(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseBenchmark, self).__init__(config)
        self.benchmark_methods = config["TESTING"]["benchmarks"]
        self.test_only = config["TESTING"].getboolean("test_only")
        self.log_test_result = config["TESTING"].getboolean("log_test_result")
        self.test_single = config["TESTING"].getboolean("test_single")
        self.test_psnr = config["TESTING"].getboolean("test_psnr")
        self.test_ssim = config["TESTING"].getboolean("test_ssim")
        self.test_visualize = config["TESTING"].getboolean("test_visualize")
        self.test_image_save = config["TESTING"].getboolean("test_image_save")


class ParseExperiment(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseExperiment, self).__init__(config)

        self.training = config["TRAINING"].getboolean("training")
        self.epoch_num = int(config["TRAINING"]["epoch_num"])
        self.skip_thr = float(config["TRAINING"]["skip_thr"])
        self.image_range = int(config["TRAINING"]["image_range"])
        self.log_every = float(config["TRAINING"]["log_every"])
        self.validate_every = float(config["TRAINING"]["validate_every"])
        self.log_psnr = config["TRAINING"].getboolean("log_psnr")
        self.log_ssim = config["TRAINING"].getboolean("log_ssim")
        self.save_path = self.experiment_save_path + "checkpoints"
        self.pre_train = config["TRAINING"]["pre_train"]
        self.only_body = config["TRAINING"].getboolean("only_body")
        self.fine_tuning = config["TRAINING"].getboolean("fine_tuning")
        self.freeze_initial_layers = config["TRAINING"].getboolean("freeze_initial_layers")
        self.chop = config["TRAINING"].getboolean("chop")
        self.save_models = config["TRAINING"].getboolean("save_models")

        if self.pre_train in ["load_latest", "load_best"]:
            self.resume = -1  # -1 --> load latest, 0--> load pre-trained model , else --> load from desired epoch
        else:
            self.resume = 0


class ParseLog(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseLog, self).__init__(config)

        if config["CHECKPOINT"].getboolean("load"):
            self.load = "./ckpts/" + self.model + "x{}".format(self.scale) + "/" + self.experiment_name
        else:
            self.load = ''
        if config["CHECKPOINT"].getboolean("save"):
            self.save = "./ckpts/" + self.model + "x{}".format(self.scale) + "/" + self.experiment_name
        else:
            self.save = ''

        self.reset = config["CHECKPOINT"].getboolean("reset")
        self.data_test = config["CHECKPOINT"]["data_test"]


class ParseMethod(ParseCommons):
    def __init__(self, config: ConfigParser):
        super(ParseMethod, self).__init__(config)

        self.learning_rate      = float(config["OPTIMIZATION"]["learning_rate"])
        self.decay              = config["OPTIMIZATION"]["decay"]
        self.decay_factor_gamma = float(config["OPTIMIZATION"]["decay_factor_gamma"])
        self.optimizer          = config["OPTIMIZATION"]["optimizer"]
        self.momentum           = float(config["OPTIMIZATION"]["momentum"])
        self.betas              = list(map(float, config["OPTIMIZATION"]["betas"].split(',')))
        self.epsilon            = float(config["OPTIMIZATION"]["epsilon"])
        self.weight_decay       = float(config["OPTIMIZATION"]["weight_decay"])
        self.gclip              = float(config["OPTIMIZATION"]["gclip"])
