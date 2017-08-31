import pickle
from optparse import OptionParser

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-i", "--test_path", dest="test_path", help="Path to test data.")
    parser.add_option("-n", "--shoe_num", dest="shoe_num", help="shoe number.")
    parser.add_option("-c", "--score", dest="score", help="score", default=99)
    (options, args) = parser.parse_args()
    if not options.test_path or not options.shoe_num:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --test_path --shoe_num to command line')
    #print options

    ## Load config object
    with open(options.test_path+'/result.pkl', 'rb') as f_in:
        result = pickle.load(f_in)

    for img in result :
	if 'boxes' in img and 'shoes' in img['boxes']:
		#print img
		box_num = len(img['boxes']['shoes'])
		if box_num<=int(options.shoe_num) and box_num > 0 :
			need_print = True
			for i in range(0,box_num):
				if img['boxes']['shoes'][i][1] < int(options.score):
					need_print = False
			if need_print:
				for i in range(0,box_num):
					print "{},{},{},{},{}".format(img['img'],img['boxes']['shoes'][i][0][0],img['boxes']['shoes'][i][0][1],img['boxes']['shoes'][i][0][2],img['boxes']['shoes'][i][0][3])

