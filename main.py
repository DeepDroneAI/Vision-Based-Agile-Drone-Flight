from pose_sampler_gt import *
import time
import argparse
parser = argparse.ArgumentParser(description='CHP')
parser.add_argument('--v_avg', default=5, type=int,
                    help='avagare velocity (1 to 5)')
args = parser.parse_args()
mode = "FLY"

def main():
	ti=time.time()
	pose_sampler = PoseSampler(v_avg=args.v_avg)
	pose_sampler.update(mode)
	


if __name__ == '__main__':
	main()

