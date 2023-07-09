import argparse


class Config_test:
    """This class defines options used during  inference  time.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--src', required=True, help='path to source image')
        parser.add_argument('--dest', required=True, help='path to subfolder trainB with output image name')
        parser.add_argument('--male', action='store_true', help = "is a man's selfie?")
        parser.add_argument("--PATH_G_A_F", type=str, default="weights/selfie2anime/GenA_F.pth.tar",
                            help="Path to female Generator")
        parser.add_argument("--PATH_G_A_F_m", type=str, default="weights/male2anime/GenA_F.pth.tar",
                            help="Path to male Generator")
        parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])


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