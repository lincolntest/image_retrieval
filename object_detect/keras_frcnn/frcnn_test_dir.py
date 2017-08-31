from frcnn import *
import pickle
import os

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--test_path", dest="test_path", help="Path to test data.")
    parser.add_option("-o", "--out_path", dest="out_path", help="Path to test data.")
    (options, args) = parser.parse_args()
    if not options.test_path or not options.out_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --test_path --out_path to command line')

    os.mkdir(options.out_path)

    ## Load config object
    with open('./config.pickle', 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    model_rpn, model_classifier, model_classifier_only = get_models(C)
    class_mapping = C.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    class_mapping = {v: k for k, v in class_mapping.items()}
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    total_boxes = []

    for idx, img_name in enumerate(sorted(os.listdir(options.test_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        filepath = os.path.join(options.test_path,img_name)
        img = cv2.imread(filepath)

	if type(img) == type(None): 
		print "find bad img"
		continue
	
        st = time.time()
	boxes_export=[]
        img,boxes_export  = detect_predict(img, C, model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color, True, False)
	total_boxes.append({'img':img_name,'boxes':boxes_export})
	print(boxes_export)
        print('Elapsed time = {}'.format(time.time() - st))
        #plt.imshow(img);plt.show()
        cv2.imwrite(options.out_path+'/result_{}.png'.format(img_name.replace('.png', '')),img)

	if idx%10 == 0:
    		pickle.dump(total_boxes, open(options.out_path+"/result.pkl", "w"))
