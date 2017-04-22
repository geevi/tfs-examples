import tensorflow as tf
from tfs import *
from image_net import *
import models

flags.DEFINE_string('train_path', '/datasets/ImageNet2012ResizedTo256/train/', "Imagenet train folder")
flags.DEFINE_string('val_path', '/datasets/ImageNet2012ResizedTo256/val/', "Imagenet val folder")
flags.DEFINE_string('class_names', 'classes.json', "Human readable names of classes.")
flags.DEFINE_string('arc', 'VGG16', "Model class name.")


def main(_):

    FLAGS.project = "imgnet"

    img_net = ImageNet( train_path  = FLAGS.train_path,  
                        val_path    = FLAGS.val_path,
                        class_names = FLAGS.class_names)

    model = find_class_by_name(FLAGS.arc, [models])(img_net)

    ctrl = init_tf(coord = True, saver = True, writer = True)

    training_loop(ctrl, model, test=True)
    

if __name__ == "__main__":
    tf.app.run()


