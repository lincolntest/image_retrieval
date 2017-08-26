from frcnn import *

if __name__ == "__main__":
    sys.argv=['','--path','/data/lincolnlin/image_retrieval/db/badcase']
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
    (options, args) = parser.parse_args()
    if not options.test_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

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

    f = open('result.csv','a')
    f.write("file,class,x1,y1,x2,y2,score\n")

    for idx, img_name in enumerate(sorted(os.listdir(options.test_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        filepath = os.path.join(options.test_path,img_name)
        img = cv2.imread(filepath)
        st = time.time()
	boxes_export=[]
        img,boxes_export  = detect_predict(img, C, model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color, True, False)
	print(boxes_export)
#	if 'shoe' in boxes_export:
#		item=boxes_export['shoe']
#		f.write('{},{},{},{},{},{},{}\n'.format(shoe[
        print('Elapsed time = {}'.format(time.time() - st))
        #plt.imshow(img);plt.show()
        cv2.imwrite('./frcnn_test_dir/result_{}.png'.format(img_name.replace('.png', '')),img)
