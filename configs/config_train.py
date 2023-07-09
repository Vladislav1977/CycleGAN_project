import argparse


class Config:
    """This class defines options used during  training  time.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--PATH_A',  help='path to subfolder trainA')
        parser.add_argument('--PATH_B', help='path to subfolder trainB')
        parser.add_argument('--LOAD', action='store_true', help='Whether to load models and optimizer params')
        parser.add_argument("--PATH_D_A", type=str, help="Path to DiscriminatorA")
        parser.add_argument("--PATH_D_F", type=str, help="Path to DiscriminatorF")
        parser.add_argument("--PATH_G_A_F", type=str, default="weights/selfie2anime/GenA_F.pth.tar",
                            help="Path to GeneratorG_A_F")
        parser.add_argument("--PATH_G_F_A", type=str, help="Path to GeneratorG_F_A")
        parser.add_argument('--SAVE', action='store_true', help='Save train params')

        parser.add_argument('--BATCH', type=int, default=1, help="Batch size")
        parser.add_argument('--lr', type=int, default=0.0002)
        parser.add_argument('--LAMBDA_A', type=int, default=10)
        parser.add_argument('--LAMBDA_B', type=int, default=10)
        parser.add_argument('--LAMBDA_IDT', type=int, default=0)
        parser.add_argument('--penalty', action='store_true',
                            help="gradient penalty term from Wasserstein loss")

        parser.add_argument("--epoches", type=int, default=100, help="Number of epoches to train")
        parser.add_argument("--epoch_count", type=int, default=100,
                            help="count the number of model trained epoches or point to start lr decay ICO train for one time")
        parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

        self.initialized = True
        return parser

    def gather_options(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()

        self.opt = opt
        return self.opt