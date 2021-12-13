from pose_sampler_loop import *
import time
import argparse
parser = argparse.ArgumentParser(description='CHP')
parser.add_argument('--v_avg', default=5, type=int,
                    help='avagare velocity (1 to 5)')
parser.add_argument("--vel_increase", default=False, type=bool, help="run with increasing speed")
args = parser.parse_args()
mode = "FLY"


def main():
	ti=time.time()
	pose_sampler = PoseSampler(v_avg=args.v_avg)
	#pose_sampler = PoseSampler(v_avg=args.v_avg, velInc=args.vel_increase)
	pose_sampler.update(mode)
	

if __name__ == '__main__':
	main()

