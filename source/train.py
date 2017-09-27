from skimage.data import imread
import numpy as np
import io
import tensorflow as tf
from multi_process_read_data import auto_load_three_sets
from multi_process_read_data import get_batches
import time






def build_cnn(batch_size):
    images = tf.Placeholder(shape=[batch_size, 180, 180, 3], type=tf.float32)
    labels = tf.Placeholder(shape=[batch_size],type=tf.float32)
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

if __name__ == '__main__':
    path = 'F:/train.bson'
    ids, imgs, cats, ws, = auto_load_three_sets(path,10)
    t1 = time.time()
    for it,(id,x,y,w) in enumerate(get_batches(ids['train'], imgs['train'], cats['train'], ws['train'], batch_size=2)):
        if it == 0:
            print(x[0])
        all_images = decode_batch_imgs(x, batch_size=2)
    t2 = time.time()




    print("Decode all images, time usage: "+str(t2-t1)+'s')
