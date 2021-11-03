ROOT = "/Users/shawnsidbon/Documents/Learning/projects-venv/HandPose"
PATH_TRAIN = ROOT + "/FreiHAND_pub_v2/training/rgb"
PATH_EVAL = ROOT + "/FreiHAND_pub_v2/evaluation/rgb"
PATH_TRAIN_XYZ = ROOT + "/FreiHAND_pub_v2/training_xyz.json"
PATH_EVAL_XYZ = ROOT + "/FreiHAND_pub_v2/evaluation_xyz.json"
PATH_TRAIN_K = ROOT + "/FreiHAND_pub_v2/training_K.json"
PATH_EVAL_K = ROOT + "/FreiHAND_pub_v2/evaluation_K.json"
PATH_TRAIN_MANO = ROOT + "/FreiHAND_pub_v2/training_mano.json"
PATH_EVAL_MANO = ROOT + "/FreiHAND_pub_v2/evaluation_mano.json"
PATH_ATTEMPT = ROOT + "/FreiHAND_pub_v2/training/rgb/00000000.jpg"
PATH_SAVED = ROOT + '/weights/cp.ckpt'

batch_size = 16
img_height = 128
img_width = 128
original_height = 224
original_width = 224
position_scaling = img_height/original_height
img_scaling = 255
workers = 4
n_imgs = 32560
range_imgs_train = list(range(int(n_imgs*0.9)))
range_imgs_test = list(range(int(n_imgs*0.9), n_imgs))
#n_imgs = 64
training_size = int(0.9*n_imgs)
MEAN = [0.40367383, 0.45234624, 0.4533549]
STD = [0.23794132, 0.21451117, 0.21778029]

